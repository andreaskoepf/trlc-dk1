#pragma once

#include <string>

namespace trlc {

// Check if running on a PREEMPT_RT kernel
bool detect_rt_kernel();

// Apply real-time scheduling. Returns true if RT was successfully applied.
// priority: SCHED_FIFO priority (1-99, higher = more priority)
// cpu: CPU core to pin to (-1 = no pinning)
// do_mlockall: lock all memory to prevent page faults
bool apply_rt_scheduling(int priority = 80, int cpu = -1, bool do_mlockall = true);

} // namespace trlc
