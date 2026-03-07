# Real-Time Scheduling in the C++ Control Loop

The C++ RT control loop (`csrc/control_loop.cpp`) runs a dedicated thread at a
fixed frequency (default 250 Hz). To achieve low-jitter timing it uses
platform-specific real-time scheduling features.

## Feature Overview

| Feature                  | Linux                              | macOS                                  |
| ------------------------ | ---------------------------------- | -------------------------------------- |
| RT scheduling            | `SCHED_FIFO` (hard RT priority)    | `THREAD_TIME_CONSTRAINT_POLICY` (Mach) |
| CPU affinity             | `pthread_setaffinity_np` (hard)    | `THREAD_AFFINITY_POLICY` (soft hint)   |
| Memory locking           | `mlockall(MCL_CURRENT\|MCL_FUTURE)` | `mlockall` (may silently fail)         |
| Precise sleep            | `clock_nanosleep` (absolute)       | `mach_wait_until` (absolute)           |
| RT kernel detection      | `/sys/kernel/realtime` + uname     | Always returns `false`                 |

## Configuration

All RT parameters are fields on `RtLoopConfig`, exposed to Python via nanobind:

```python
from trlc_dk1_control._trlc_dk1_rt import RtLoopConfig

cfg = RtLoopConfig()
cfg.loop_hz          = 250.0   # Control loop frequency
cfg.rt_priority      = 80      # SCHED_FIFO priority 1-99 (Linux only)
cfg.rt_cpu_affinity  = -1      # CPU core to pin to (-1 = no pinning)
cfg.rt_use_mlockall  = True    # Lock all memory to prevent page faults
```

### `loop_hz`

The target frequency of the control loop. On macOS this value is also used to
configure the Mach time constraint policy (period, computation budget, and
constraint window).

### `rt_priority` (Linux only)

The `SCHED_FIFO` real-time priority, range 1–99 (higher = more priority).
Default is 80. This is ignored on macOS which uses
`THREAD_TIME_CONSTRAINT_POLICY` instead.

### `rt_cpu_affinity`

Pin the RT thread to a specific CPU core. Default `-1` means no pinning (the OS
schedules the thread on any available core).

**Linux:** Uses `pthread_setaffinity_np` for hard pinning — the thread will only
run on the specified core.

**macOS:** Uses Mach `THREAD_AFFINITY_POLICY` affinity tags. This is a soft
hint — threads with different tags are *encouraged* to run on different cores,
but the scheduler may override this.

**Bimanual setups:** When running two `RtControlLoop` instances (left and right
arm), assign different CPU cores to avoid contention:

```python
left_cfg.rt_cpu_affinity = 2
right_cfg.rt_cpu_affinity = 3
```

With the default `-1`, both loops are scheduled freely by the OS, which works
fine in most cases.

### `rt_use_mlockall`

When `True` (the default), calls `mlockall(MCL_CURRENT | MCL_FUTURE)` to lock
all current and future memory pages, preventing page faults in the hot loop.
Requires `CAP_IPC_LOCK` or sufficient `RLIMIT_MEMLOCK` on Linux.

## Linux Setup

### Permissions for SCHED_FIFO

By default, unprivileged users cannot set `SCHED_FIFO`. There are several ways
to grant access:

**Option 1: `rtprio` via `/etc/security/limits.conf`** (recommended)

Add a line for your user or group:

```
# /etc/security/limits.conf
@realtime  -  rtprio  99
@realtime  -  memlock unlimited
```

Then add your user to the `realtime` group:

```bash
sudo groupadd -f realtime
sudo usermod -aG realtime $USER
```

Log out and back in for the changes to take effect.

**Option 2: Per-binary capability**

Grant the RT capability to the Python interpreter:

```bash
sudo setcap cap_sys_nice,cap_ipc_lock+ep $(readlink -f .venv/bin/python)
```

Note: This grants RT scheduling to *all* Python programs run with this
interpreter. The capability is lost when the binary is updated.

**Option 3: Run as root**

```bash
sudo uv run examples/test_rt_loop.py
```

Not recommended for regular use.

### PREEMPT_RT Kernel

For the best real-time guarantees, install a `PREEMPT_RT`-patched kernel. The
control loop will detect this automatically via `detect_rt_kernel()`.

On Ubuntu/Debian:

```bash
sudo apt install linux-image-rt-amd64    # Debian
sudo apt install linux-lowlatency        # Ubuntu (close but not full RT)
```

For a full PREEMPT_RT kernel on Ubuntu, check the
[Ubuntu Pro real-time kernel](https://ubuntu.com/real-time) or build from
source with the RT patch from [kernel.org](https://wiki.linuxfoundation.org/realtime/start).

After installation, reboot and verify:

```bash
uname -r   # Should contain PREEMPT_RT or preempt_rt
cat /sys/kernel/realtime   # Should print 1
```

The control loop works without a PREEMPT_RT kernel — `SCHED_FIFO` still
provides priority scheduling — but worst-case latencies will be higher due to
non-preemptible kernel code paths.

### Kernel Tuning

For lowest jitter on an RT kernel, consider:

```bash
# Isolate CPU cores 2-3 from the general scheduler (add to GRUB_CMDLINE_LINUX)
isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3

# Then pin the control loop to an isolated core
cfg.rt_cpu_affinity = 2
```

Disable CPU frequency scaling on the pinned cores:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
echo performance | sudo tee /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
```

## macOS Setup

### THREAD_TIME_CONSTRAINT_POLICY

On macOS, the control loop uses the Mach `THREAD_TIME_CONSTRAINT_POLICY`. This
is the same mechanism used by CoreAudio and other low-latency frameworks. It
does **not require root** — any user process can request it.

The policy parameters are derived from `loop_hz`:

| Parameter     | Value              | Meaning                                   |
| ------------- | ------------------ | ----------------------------------------- |
| `period`      | `1/loop_hz`        | Expected interval between wakeups         |
| `computation` | `period / 2` (50%) | Maximum CPU time needed per cycle         |
| `constraint`  | `period * 4/5` (80%) | Deadline: must finish within this window |
| `preemptible` | `true`             | Can be preempted if computation is exceeded |

### Limitations

- No `PREEMPT_RT` equivalent exists on macOS — worst-case latencies are higher
  than a properly configured Linux RT system.
- CPU affinity is a soft hint, not a hard guarantee.
- `mlockall` may silently fail or have no effect on macOS.
- macOS is suitable for development and testing but Linux with PREEMPT_RT is
  recommended for production robot control.

## Graceful Degradation

All RT features fail gracefully. If `SCHED_FIFO`, `mlockall`, CPU affinity, or
`THREAD_TIME_CONSTRAINT_POLICY` fail (e.g. due to missing permissions), the
control loop prints a warning to stderr and continues with the default
scheduler. The `is_rt_active()` method returns whether RT scheduling was
successfully applied:

```python
loop = RtControlLoop(cfg)
loop.start()
print(f"RT active: {loop.is_rt_active()}")
```
