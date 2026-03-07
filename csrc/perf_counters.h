#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <time.h>

namespace trlc {

struct PerfSnapshot {
    uint64_t loop_count = 0;
    double min_cycle_us = std::numeric_limits<double>::max();
    double max_cycle_us = 0.0;
    double mean_cycle_us = 0.0;
    uint64_t deadline_misses = 0;
    std::array<uint64_t, 6> histogram = {};  // [0-100us, 100-500us, 500-1ms, 1-2ms, 2-4ms, >4ms]
};

class PerfCounters {
public:
    PerfCounters() = default;

    // Called by RT thread each cycle
    void record(double cycle_us, double target_us) {
        // Handle reset request from non-RT thread (deferred to RT thread to avoid data race)
        if (reset_requested_.load(std::memory_order_acquire)) {
            reset_requested_.store(false, std::memory_order_relaxed);
            stats_ = PerfSnapshot{};
            sum_ = 0.0;
            uint64_t s = seq_.load(std::memory_order_relaxed);
            seq_.store(s + 1, std::memory_order_release);
            snap_ = PerfSnapshot{};
            seq_.store(s + 2, std::memory_order_release);
            ring_head_.store(0, std::memory_order_release);
            reset_ack_.fetch_add(1, std::memory_order_release);
        }

        // Update running stats
        ++stats_.loop_count;
        if (cycle_us < stats_.min_cycle_us) stats_.min_cycle_us = cycle_us;
        if (cycle_us > stats_.max_cycle_us) stats_.max_cycle_us = cycle_us;
        sum_ += cycle_us;
        stats_.mean_cycle_us = sum_ / static_cast<double>(stats_.loop_count);

        if (cycle_us > target_us * 1.5) ++stats_.deadline_misses;

        // Histogram
        int bin;
        if (cycle_us < 100.0)       bin = 0;
        else if (cycle_us < 500.0)  bin = 1;
        else if (cycle_us < 1000.0) bin = 2;
        else if (cycle_us < 2000.0) bin = 3;
        else if (cycle_us < 4000.0) bin = 4;
        else                        bin = 5;
        ++stats_.histogram[static_cast<size_t>(bin)];

        // Write snapshot under seqlock
        uint64_t s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_release);  // odd = writing
        snap_ = stats_;
        seq_.store(s + 2, std::memory_order_release);  // even = done

        // Write to ring buffer
        ring_[ring_head_.load(std::memory_order_relaxed) % RING_SIZE] = static_cast<float>(cycle_us);
        ring_head_.fetch_add(1, std::memory_order_release);
    }

    // Called from Python thread (non-blocking)
    PerfSnapshot snapshot() const {
        PerfSnapshot result;
        for (int attempt = 0; attempt < 100; ++attempt) {
            uint64_t s1 = seq_.load(std::memory_order_acquire);
            if (s1 & 1) continue;  // writer in progress
            result = snap_;
            uint64_t s2 = seq_.load(std::memory_order_acquire);
            if (s1 == s2) return result;
        }
        return result;  // best effort
    }

    // Copy recent cycle times into caller buffer, return count copied
    size_t read_ring(float* out, size_t max_count) const {
        for (int attempt = 0; attempt < 10; ++attempt) {
            uint64_t head = ring_head_.load(std::memory_order_acquire);
            uint64_t avail = std::min(head, static_cast<uint64_t>(RING_SIZE));
            size_t count = std::min(static_cast<size_t>(avail), max_count);
            uint64_t start = head - count;

            for (size_t i = 0; i < count; ++i) {
                out[i] = ring_[(start + i) % RING_SIZE];
            }

            // Verify head didn't advance too much (data wasn't overwritten)
            uint64_t head2 = ring_head_.load(std::memory_order_acquire);
            if (head2 - head < RING_SIZE / 2) {
                return count;
            }
            // Retry if writer lapped us
        }
        return 0;
    }

    // Request reset (processed by RT thread on next record() call).
    // Blocks until the RT thread has processed the reset, with a timeout.
    // Throws std::runtime_error if the RT thread does not process the reset in time.
    void reset(int timeout_ms = 100) {
        uint64_t before = reset_ack_.load(std::memory_order_acquire);
        reset_requested_.store(true, std::memory_order_release);
        if (timeout_ms <= 0) return;  // fire-and-forget
        for (int waited = 0; waited < timeout_ms; ++waited) {
            if (reset_ack_.load(std::memory_order_acquire) != before) return;
            struct timespec ts = {0, 1000000};  // 1ms
            nanosleep(&ts, nullptr);
        }
        throw std::runtime_error("PerfCounters::reset() timed out waiting for RT thread acknowledgment");
    }

private:
    // Writer-side running stats (only touched by RT thread)
    PerfSnapshot stats_;
    double sum_ = 0.0;

    // Seqlock-protected snapshot
    alignas(64) PerfSnapshot snap_;
    alignas(64) std::atomic<uint64_t> seq_{0};

    // Ring buffer
    static constexpr size_t RING_SIZE = 16384;
    alignas(64) std::array<float, RING_SIZE> ring_{};
    alignas(64) std::atomic<uint64_t> ring_head_{0};

    // Reset coordination (Python sets, RT thread consumes and acks)
    std::atomic<bool> reset_requested_{false};
    std::atomic<uint64_t> reset_ack_{0};
};

} // namespace trlc
