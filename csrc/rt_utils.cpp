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

#ifdef __APPLE__
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#endif

namespace trlc {

bool detect_rt_kernel() {
#ifdef __APPLE__
    // macOS has no PREEMPT_RT; always return false
    return false;
#else
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
#endif
}

bool apply_rt_scheduling(int priority, int cpu, bool do_mlockall) {
    bool success = true;

    // Set SCHED_FIFO (available on both Linux and macOS)
    struct sched_param param{};
    param.sched_priority = priority;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        std::fprintf(stderr, "rt_utils: SCHED_FIFO priority %d failed: %s\n",
                     priority, std::strerror(errno));
        success = false;
    }

    // CPU affinity
    if (cpu >= 0) {
#ifdef __APPLE__
        // macOS uses thread affinity tags (hints, not hard pinning)
        thread_affinity_policy_data_t policy = { cpu + 1 };
        kern_return_t kr = thread_policy_set(
            pthread_mach_thread_np(pthread_self()),
            THREAD_AFFINITY_POLICY,
            reinterpret_cast<thread_policy_t>(&policy),
            THREAD_AFFINITY_POLICY_COUNT);
        if (kr != KERN_SUCCESS) {
            std::fprintf(stderr, "rt_utils: thread affinity tag %d failed: %d\n", cpu, kr);
            success = false;
        }
#else
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu, &cpuset);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
            std::fprintf(stderr, "rt_utils: CPU affinity to core %d failed: %s\n",
                         cpu, std::strerror(errno));
            success = false;
        }
#endif
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
