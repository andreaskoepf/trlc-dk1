#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "dm_protocol.h"
#include "gravity_comp.h"
#include "motor_defs.h"
#include "perf_counters.h"
#include "safety.h"
#include "serial_port.h"

namespace trlc {

struct RtLoopConfig {
    std::string serial_port = "/dev/ttyACM0";
    double loop_hz = 250.0;

    std::vector<MotorDescriptor> motors;  // 7 entries (6 arm + gripper)

    std::array<double, 6> default_kp = {80, 70, 60, 20, 20, 10};
    std::array<double, 6> default_kd = {5, 5, 4, 1, 1, 1};

    // Flat [lo0,hi0,lo1,hi1,...] for 6 joints
    std::array<double, 12> joint_pos_limits = {
        -3.14159265, 3.14159265,   // joint_1
        -3.14159265, 3.14159265,   // joint_2
        -3.14159265, 3.14159265,   // joint_3
        -1.74532925, 1.74532925,   // joint_4 (100 deg)
        -1.57079633, 1.57079633,   // joint_5 (90 deg)
        -3.14159265, 3.14159265,   // joint_6
    };
    std::array<double, 6> joint_torque_limits = {28, 28, 28, 10, 10, 10};
    double limit_buffer = 0.05;

    std::string model_path;  // URDF/XML for gravity comp, empty = disabled
    double gravity_comp_scale = 1.0;

    double command_timeout_s = 0.5;
    int overcurrent_threshold = 20;

    // Gripper
    double gripper_open_pos = 0.0;
    double gripper_closed_pos = -4.7;
    double max_gripper_torque_nm = 1.0;
    double torque_constant = 0.945;
    double emit_velocity_scale = 100.0;
    double emit_current_scale = 1000.0;

    // RT
    int rt_priority = 80;
    int rt_cpu_affinity = -1;
    bool rt_use_mlockall = true;
};

struct JointState {
    std::array<double, 6> pos = {};
    std::array<double, 6> vel = {};
    std::array<double, 6> torque = {};
};

struct GripperState {
    double pos = 0.0;
    double torque = 0.0;
};

class RtControlLoop {
public:
    explicit RtControlLoop(const RtLoopConfig& cfg);
    ~RtControlLoop();

    RtControlLoop(const RtControlLoop&) = delete;
    RtControlLoop& operator=(const RtControlLoop&) = delete;

    void start();
    void stop();

    // Commands (lock-free writes via seqlock)
    void command_joint_pos(const double* q6);
    void command_gripper(double normalized);  // 0=open, 1=closed

    // State (lock-free reads via seqlock)
    JointState get_joint_state() const;
    GripperState get_gripper_state() const;

    // Diagnostics
    PerfSnapshot get_perf() const;
    size_t read_cycle_times(float* buf, size_t max) const;
    void reset_perf();
    bool is_running() const { return running_.load(std::memory_order_acquire); }
    bool is_rt_active() const { return rt_active_; }

private:
    void configure_motors();
    void calibrate_gripper();
    void rt_thread_func();

    // Seqlock helpers
    struct CommandBuffer {
        std::array<double, 6> q_des = {};
        double gripper_des = 0.0;
        uint64_t timestamp_ns = 0;
    };

    struct StateBuffer {
        std::array<double, 7> pos = {};
        std::array<double, 7> vel = {};
        std::array<double, 7> torque = {};
    };

    RtLoopConfig cfg_;
    SerialPort serial_;
    GravityCompensator grav_comp_;
    PacketParser parser_;
    PerfCounters perf_;
    SafetyState safety_state_;

    // Command seqlock (Python writes, RT reads)
    alignas(64) CommandBuffer cmd_buf_;
    alignas(64) std::atomic<uint64_t> cmd_seq_{0};

    // State seqlock (RT writes, Python reads)
    alignas(64) StateBuffer state_buf_;
    alignas(64) std::atomic<uint64_t> state_seq_{0};

    std::atomic<bool> running_{false};
    std::thread thread_;
    bool rt_active_ = false;

    // Gripper calibration result
    double gripper_open_pos_ = 0.0;
};

} // namespace trlc
