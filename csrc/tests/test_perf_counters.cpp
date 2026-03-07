#include <cassert>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>
#include <atomic>

#include "perf_counters.h"

using namespace trlc;

static void test_basic_recording() {
    PerfCounters pc;

    pc.record(100.0, 4000.0);
    pc.record(200.0, 4000.0);
    pc.record(300.0, 4000.0);

    auto snap = pc.snapshot();
    assert(snap.loop_count == 3);
    assert(std::abs(snap.min_cycle_us - 100.0) < 0.01);
    assert(std::abs(snap.max_cycle_us - 300.0) < 0.01);
    assert(std::abs(snap.mean_cycle_us - 200.0) < 0.01);
    assert(snap.deadline_misses == 0);  // none exceed 6000us

    std::printf("  basic recording: PASS\n");
}

static void test_deadline_miss() {
    PerfCounters pc;

    pc.record(100.0, 4000.0);
    pc.record(7000.0, 4000.0);  // > 6000 = 1.5x target

    auto snap = pc.snapshot();
    assert(snap.deadline_misses == 1);

    std::printf("  deadline miss: PASS\n");
}

static void test_histogram() {
    PerfCounters pc;

    pc.record(50.0, 4000.0);     // bin 0: 0-100us
    pc.record(250.0, 4000.0);    // bin 1: 100-500us
    pc.record(750.0, 4000.0);    // bin 2: 500-1ms
    pc.record(1500.0, 4000.0);   // bin 3: 1-2ms
    pc.record(3000.0, 4000.0);   // bin 4: 2-4ms
    pc.record(5000.0, 4000.0);   // bin 5: >4ms

    auto snap = pc.snapshot();
    assert(snap.histogram[0] == 1);
    assert(snap.histogram[1] == 1);
    assert(snap.histogram[2] == 1);
    assert(snap.histogram[3] == 1);
    assert(snap.histogram[4] == 1);
    assert(snap.histogram[5] == 1);

    std::printf("  histogram: PASS\n");
}

static void test_ring_buffer() {
    PerfCounters pc;

    for (int i = 0; i < 100; ++i) {
        pc.record(static_cast<double>(i), 4000.0);
    }

    float buf[200];
    size_t n = pc.read_ring(buf, 200);
    assert(n == 100);

    // Last element should be 99
    assert(std::abs(buf[n - 1] - 99.0f) < 0.01f);
    // First element should be 0
    assert(std::abs(buf[0] - 0.0f) < 0.01f);

    std::printf("  ring buffer: PASS\n");
}

static void test_ring_buffer_overflow() {
    PerfCounters pc;

    // Write more than RING_SIZE entries
    for (int i = 0; i < 20000; ++i) {
        pc.record(static_cast<double>(i), 4000.0);
    }

    float buf[16384];
    size_t n = pc.read_ring(buf, 16384);
    assert(n == 16384);

    // Last entry should be 19999
    assert(std::abs(buf[n - 1] - 19999.0f) < 0.01f);

    std::printf("  ring buffer overflow: PASS\n");
}

static void test_concurrent_access() {
    PerfCounters pc;
    std::atomic<bool> running{true};
    constexpr int NUM_WRITES = 100000;

    // Writer thread (simulates RT thread)
    std::thread writer([&]() {
        for (int i = 0; i < NUM_WRITES; ++i) {
            pc.record(static_cast<double>(i % 4000), 4000.0);
        }
        running.store(false);
    });

    // Reader thread (simulates Python thread)
    int read_count = 0;
    while (running.load()) {
        auto snap = pc.snapshot();
        (void)snap;

        float buf[100];
        pc.read_ring(buf, 100);

        ++read_count;
    }

    writer.join();

    auto snap = pc.snapshot();
    assert(snap.loop_count == NUM_WRITES);
    assert(read_count > 0);

    std::printf("  concurrent access (%d reads during %d writes): PASS\n",
                read_count, NUM_WRITES);
}

static void test_reset() {
    PerfCounters pc;

    pc.record(100.0, 4000.0);
    pc.record(200.0, 4000.0);

    pc.reset(0);  // non-blocking (no RT thread in unit test)
    // Reset is deferred — the next record() call processes it, then records
    pc.record(50.0, 4000.0);

    auto snap = pc.snapshot();
    // After reset + one new record: loop_count=1, min=50, max=50
    assert(snap.loop_count == 1);
    assert(snap.deadline_misses == 0);
    assert(std::abs(snap.min_cycle_us - 50.0) < 0.01);
    assert(std::abs(snap.max_cycle_us - 50.0) < 0.01);

    std::printf("  reset: PASS\n");
}

int main() {
    std::printf("perf_counters tests:\n");
    test_basic_recording();
    test_deadline_miss();
    test_histogram();
    test_ring_buffer();
    test_ring_buffer_overflow();
    test_concurrent_access();
    test_reset();
    std::printf("All perf_counters tests passed!\n");
    return 0;
}
