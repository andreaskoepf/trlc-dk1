#include "rt_utils.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/utsname.h>

namespace trlc {

bool detect_rt_kernel() {
    // Method 1: check /sys/kernel/realtime
    {
        std::ifstream f("/sys/kernel/realtime");
        if (f.is_open()) {
            int val = 0;
            f >> val;
            if (val == 1) return true;
        }
    }

    // Method 2: check uname -r for PREEMPT_RT
    {
        struct utsname uts{};
        if (uname(&uts) == 0) {
            std::string release(uts.release);
            if (release.find("PREEMPT_RT") != std::string::npos ||
                release.find("preempt_rt") != std::string::npos) {
                return true;
            }
        }
    }

    return false;
}

bool apply_rt_scheduling(int priority, int cpu, bool do_mlockall) {
    bool success = true;

    // Set SCHED_FIFO
    struct sched_param param{};
    param.sched_priority = priority;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        std::fprintf(stderr, "rt_utils: SCHED_FIFO priority %d failed: %s\n",
                     priority, std::strerror(errno));
        success = false;
    }

    // CPU affinity
    if (cpu >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu, &cpuset);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
            std::fprintf(stderr, "rt_utils: CPU affinity to core %d failed: %s\n",
                         cpu, std::strerror(errno));
            success = false;
        }
    }

    // mlockall
    if (do_mlockall) {
        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
            std::fprintf(stderr, "rt_utils: mlockall failed: %s\n", std::strerror(errno));
            success = false;
        }
    }

    return success;
}

} // namespace trlc
