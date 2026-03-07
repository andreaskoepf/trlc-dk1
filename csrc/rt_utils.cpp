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
#include <mach/mach_time.h>
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

bool apply_rt_scheduling(int priority, int cpu, bool do_mlockall, double loop_hz) {
    bool success = true;

#ifdef __APPLE__
    // Use Mach THREAD_TIME_CONSTRAINT_POLICY for real-time scheduling (no root required).
    // This tells the scheduler we are a periodic real-time thread.
    {
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        // Convert nanoseconds to mach absolute time units
        auto ns_to_abs = [&](uint64_t ns) -> uint64_t {
            return ns * tb.denom / tb.numer;
        };

        uint64_t period_ns = static_cast<uint64_t>(1e9 / loop_hz);
        // Budget ~50% of period for computation, constraint = 80% of period
        uint64_t computation_ns = period_ns / 2;
        uint64_t constraint_ns = period_ns * 4 / 5;

        thread_time_constraint_policy_data_t policy;
        policy.period      = static_cast<uint32_t>(ns_to_abs(period_ns));
        policy.computation = static_cast<uint32_t>(ns_to_abs(computation_ns));
        policy.constraint  = static_cast<uint32_t>(ns_to_abs(constraint_ns));
        policy.preemptible = 1;

        kern_return_t kr = thread_policy_set(
            pthread_mach_thread_np(pthread_self()),
            THREAD_TIME_CONSTRAINT_POLICY,
            reinterpret_cast<thread_policy_t>(&policy),
            THREAD_TIME_CONSTRAINT_POLICY_COUNT);
        if (kr != KERN_SUCCESS) {
            std::fprintf(stderr, "rt_utils: THREAD_TIME_CONSTRAINT_POLICY failed: %d\n", kr);
            success = false;
        }
    }

    // CPU affinity (soft hint via affinity tags on macOS)
    if (cpu >= 0) {
        thread_affinity_policy_data_t affinity = { cpu + 1 };
        kern_return_t kr = thread_policy_set(
            pthread_mach_thread_np(pthread_self()),
            THREAD_AFFINITY_POLICY,
            reinterpret_cast<thread_policy_t>(&affinity),
            THREAD_AFFINITY_POLICY_COUNT);
        if (kr != KERN_SUCCESS) {
            std::fprintf(stderr, "rt_utils: thread affinity tag %d failed: %d\n", cpu, kr);
            success = false;
        }
    }
#else
    // Linux: use SCHED_FIFO for real-time priority
    (void)loop_hz;
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
#endif

    // mlockall (available on both platforms)
    if (do_mlockall) {
        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
            std::fprintf(stderr, "rt_utils: mlockall failed: %s\n", std::strerror(errno));
            success = false;
        }
    }

    return success;
}

} // namespace trlc
