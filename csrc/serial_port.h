#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace trlc {

class SerialPort {
public:
    SerialPort() = default;
    ~SerialPort();

    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    bool open(const std::string& device, int baudrate = 921600);
    void close();

    // Write exactly n bytes. Returns true on success.
    bool write(const uint8_t* buf, size_t n);

    // Non-blocking read up to max bytes. Returns number of bytes read.
    size_t read_all(uint8_t* buf, size_t max);

    // Blocking read with timeout (uses select). Returns number of bytes read.
    // Waits up to timeout_us microseconds for data to become available,
    // then reads all available bytes.
    size_t read_with_timeout(uint8_t* buf, size_t max, int timeout_us);

    // Wait for all output to be transmitted (tcdrain wrapper).
    void drain();

    int fd() const { return fd_; }
    bool is_open() const { return fd_ >= 0; }

private:
    int fd_ = -1;
};

} // namespace trlc
