#include "serial_port.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

namespace trlc {

SerialPort::~SerialPort() {
    close();
}

bool SerialPort::open(const std::string& device, int baudrate) {
    if (fd_ >= 0) close();

    fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        std::fprintf(stderr, "SerialPort: cannot open %s: %s\n",
                     device.c_str(), std::strerror(errno));
        return false;
    }

    // Clear O_NONBLOCK after open (we want blocking writes, non-blocking reads via VMIN/VTIME)
    int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags & ~O_NONBLOCK);

    struct termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
        std::fprintf(stderr, "SerialPort: tcgetattr failed: %s\n", std::strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    // Map baudrate
    speed_t baud;
    switch (baudrate) {
        case 9600:    baud = B9600;    break;
        case 19200:   baud = B19200;   break;
        case 38400:   baud = B38400;   break;
        case 57600:   baud = B57600;   break;
        case 115200:  baud = B115200;  break;
        case 230400:  baud = B230400;  break;
        case 460800:  baud = B460800;  break;
        case 500000:  baud = B500000;  break;
        case 576000:  baud = B576000;  break;
        case 921600:  baud = B921600;  break;
        case 1000000: baud = B1000000; break;
        default:
            std::fprintf(stderr, "SerialPort: unsupported baudrate %d\n", baudrate);
            ::close(fd_);
            fd_ = -1;
            return false;
    }

    cfsetispeed(&tty, baud);
    cfsetospeed(&tty, baud);

    // Raw mode (8N1, no flow control)
    cfmakeraw(&tty);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    // Non-blocking read: VMIN=0, VTIME=0 -> return immediately with available data
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        std::fprintf(stderr, "SerialPort: tcsetattr failed: %s\n", std::strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    // Flush any existing data
    tcflush(fd_, TCIOFLUSH);

    return true;
}

void SerialPort::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool SerialPort::write(const uint8_t* buf, size_t n) {
    if (fd_ < 0) return false;
    size_t written = 0;
    while (written < n) {
        ssize_t r = ::write(fd_, buf + written, n - written);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        written += static_cast<size_t>(r);
    }
    return true;
}

size_t SerialPort::read_all(uint8_t* buf, size_t max) {
    if (fd_ < 0) return 0;
    ssize_t r = ::read(fd_, buf, max);
    if (r < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        return 0;
    }
    return static_cast<size_t>(r);
}

size_t SerialPort::read_with_timeout(uint8_t* buf, size_t max, int timeout_us) {
    if (fd_ < 0) return 0;

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd_, &fds);

    struct timeval tv;
    tv.tv_sec = timeout_us / 1000000;
    tv.tv_usec = timeout_us % 1000000;

    int ret = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
    if (ret <= 0) return 0;  // timeout or error

    // Data available — read all that's there
    size_t total = 0;
    while (total < max) {
        ssize_t r = ::read(fd_, buf + total, max - total);
        if (r <= 0) break;
        total += static_cast<size_t>(r);
        // Check if more data is available without blocking
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);
        tv.tv_sec = 0;
        tv.tv_usec = 0;
        if (select(fd_ + 1, &fds, nullptr, nullptr, &tv) <= 0) break;
    }
    return total;
}

void SerialPort::drain() {
    if (fd_ >= 0) {
        tcdrain(fd_);
    }
}

} // namespace trlc
