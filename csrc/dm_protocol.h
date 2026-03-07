#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "motor_defs.h"

namespace trlc {

// --- Quantization (matches DM_CAN.py float_to_uint / uint_to_float) ---

inline uint16_t float_to_uint(float x, float x_min, float x_max, int bits) {
    if (x < x_min) x = x_min;
    if (x > x_max) x = x_max;
    float span = x_max - x_min;
    float norm = (x - x_min) / span;
    return static_cast<uint16_t>(norm * ((1 << bits) - 1));
}

inline float uint_to_float(uint16_t x, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    float norm = static_cast<float>(x) / ((1 << bits) - 1);
    return norm * span + x_min;
}

// --- Send frame format ---
// 30-byte frame: header + CAN data, matches DM_CAN.py send_data_frame layout
// [0x55, 0xAA, 0x1E, 0x03, 0x01, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00,
//  id_lo, id_hi, 0, 0, 0x00, 0x08, 0x00, 0x00, data[0..7], 0x00]

static constexpr uint8_t FRAME_TEMPLATE[30] = {
    0x55, 0xAA, 0x1E, 0x03, 0x01, 0x00, 0x00, 0x00,
    0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

// Fills a 30-byte buffer with a send frame for the given motor_id and 8-byte data payload.
inline void build_send_frame(uint8_t* buf30, uint16_t motor_id, const uint8_t* data8) {
    std::memcpy(buf30, FRAME_TEMPLATE, 30);
    buf30[13] = motor_id & 0xFF;
    buf30[14] = (motor_id >> 8) & 0xFF;
    std::memcpy(buf30 + 21, data8, 8);
}

// --- MIT mode frame ---

inline void build_mit_frame(uint8_t* buf30, uint16_t slave_id,
                            float kp, float kd, float q, float dq, float tau,
                            const MotorLimits& lim) {
    uint16_t kp_uint  = float_to_uint(kp, 0.0f, 500.0f, 12);
    uint16_t kd_uint  = float_to_uint(kd, 0.0f, 5.0f, 12);
    uint16_t q_uint   = float_to_uint(q, -lim.q_max, lim.q_max, 16);
    uint16_t dq_uint  = float_to_uint(dq, -lim.dq_max, lim.dq_max, 12);
    uint16_t tau_uint = float_to_uint(tau, -lim.tau_max, lim.tau_max, 12);

    uint8_t data[8];
    data[0] = (q_uint >> 8) & 0xFF;
    data[1] = q_uint & 0xFF;
    data[2] = dq_uint >> 4;
    data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF);
    data[4] = kp_uint & 0xFF;
    data[5] = kd_uint >> 4;
    data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF);
    data[7] = tau_uint & 0xFF;

    build_send_frame(buf30, slave_id, data);
}

// --- EMIT (Torque_Pos) mode frame ---

inline void build_emit_frame(uint8_t* buf30, uint16_t slave_id,
                             float pos, float vel, float i_des) {
    // motor_id for EMIT = 0x300 + slave_id
    uint16_t motor_id = 0x300 + slave_id;

    // Position is packed as IEEE 754 float32 (little-endian)
    uint8_t data[8] = {};
    float pos_val = pos;
    std::memcpy(data, &pos_val, 4);

    uint16_t vel_uint = static_cast<uint16_t>(vel);
    uint16_t ides_uint = static_cast<uint16_t>(i_des);
    data[4] = vel_uint & 0xFF;
    data[5] = vel_uint >> 8;
    data[6] = ides_uint & 0xFF;
    data[7] = ides_uint >> 8;

    build_send_frame(buf30, motor_id, data);
}

// --- VEL mode frame ---

inline void build_vel_frame(uint8_t* buf30, uint16_t slave_id, float vel) {
    uint16_t motor_id = 0x200 + slave_id;
    uint8_t data[8] = {};
    float vel_val = vel;
    std::memcpy(data, &vel_val, 4);
    build_send_frame(buf30, motor_id, data);
}

// --- Control command frames (enable/disable/set-zero) ---

inline void build_enable_frame(uint8_t* buf30, uint16_t slave_id) {
    uint8_t data[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC};
    build_send_frame(buf30, slave_id, data);
}

inline void build_disable_frame(uint8_t* buf30, uint16_t slave_id) {
    uint8_t data[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD};
    build_send_frame(buf30, slave_id, data);
}

inline void build_set_zero_frame(uint8_t* buf30, uint16_t slave_id) {
    uint8_t data[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};
    build_send_frame(buf30, slave_id, data);
}

// --- Refresh motor status (0x7FF + 0xCC) ---

inline void build_refresh_frame(uint8_t* buf30, uint16_t slave_id) {
    uint8_t data[8] = {};
    data[0] = slave_id & 0xFF;
    data[1] = (slave_id >> 8) & 0xFF;
    data[2] = 0xCC;
    build_send_frame(buf30, 0x7FF, data);
}

// --- Switch control mode (parameter write RID=10) ---

inline void build_switch_mode_frame(uint8_t* buf30, uint16_t slave_id, uint8_t mode) {
    uint8_t data[8] = {};
    data[0] = slave_id & 0xFF;
    data[1] = (slave_id >> 8) & 0xFF;
    data[2] = 0x55;  // write command
    data[3] = 10;    // RID = CTRL_MODE
    // mode as uint32 little-endian in bytes 4..7
    data[4] = mode;
    data[5] = 0;
    data[6] = 0;
    data[7] = 0;
    build_send_frame(buf30, 0x7FF, data);
}

// --- Read parameter (RID) ---

inline void build_read_param_frame(uint8_t* buf30, uint16_t slave_id, uint8_t rid) {
    uint8_t data[8] = {};
    data[0] = slave_id & 0xFF;
    data[1] = (slave_id >> 8) & 0xFF;
    data[2] = 0x33;  // read command
    data[3] = rid;
    build_send_frame(buf30, 0x7FF, data);
}

// --- Receive packet parsing ---

static constexpr size_t RX_PACKET_LEN = 16;
static constexpr uint8_t RX_HEADER = 0xAA;
static constexpr uint8_t RX_TAIL   = 0x55;

struct MotorState {
    float q;
    float dq;
    float tau;
};

inline MotorState decode_motor_state(const uint8_t* packet, const MotorLimits& lim) {
    // Data bytes start at offset 7 in the 16-byte packet
    const uint8_t* data = packet + 7;
    uint16_t q_uint   = (static_cast<uint16_t>(data[1]) << 8) | data[2];
    uint16_t dq_uint  = (static_cast<uint16_t>(data[3]) << 4) | (data[4] >> 4);
    uint16_t tau_uint = ((data[4] & 0x0F) << 8) | data[5];

    MotorState s;
    s.q   = uint_to_float(q_uint, -lim.q_max, lim.q_max, 16);
    s.dq  = uint_to_float(dq_uint, -lim.dq_max, lim.dq_max, 12);
    s.tau = uint_to_float(tau_uint, -lim.tau_max, lim.tau_max, 12);
    return s;
}

// Extract the CAN ID from a 16-byte receive packet
inline uint32_t decode_can_id(const uint8_t* packet) {
    return static_cast<uint32_t>(packet[3]) |
           (static_cast<uint32_t>(packet[4]) << 8) |
           (static_cast<uint32_t>(packet[5]) << 16) |
           (static_cast<uint32_t>(packet[6]) << 24);
}

// Extract the slave/master ID from the data field of a response
inline uint8_t decode_response_id(const uint8_t* packet) {
    // data[0] (at offset 7) contains the motor ID info
    return packet[7] & 0x0F;
}

// CMD byte is at offset 1
inline uint8_t decode_cmd(const uint8_t* packet) {
    return packet[1];
}

// Check if a CMD=0x11 packet is a parameter response (not motor state).
// Parameter responses have data[2] == 0x33 (write ack) or 0x55 (read ack).
inline bool is_param_response(const uint8_t* packet) {
    uint8_t d2 = packet[7 + 2];  // data[2]
    return d2 == 0x33 || d2 == 0x55;
}

class PacketParser {
public:
    // Feed raw bytes, extract complete 16-byte packets
    void feed(const uint8_t* data, size_t len) {
        residual_.insert(residual_.end(), data, data + len);
        // Cap residual buffer to prevent unbounded growth from corrupted data
        if (residual_.size() > MAX_RESIDUAL) {
            size_t excess = residual_.size() - MAX_RESIDUAL;
            residual_.erase(residual_.begin(),
                           residual_.begin() + static_cast<ptrdiff_t>(excess));
        }
    }

    // Extract all complete packets from the residual buffer
    // Returns number of packets extracted
    size_t extract(std::vector<std::array<uint8_t, RX_PACKET_LEN>>& out) {
        size_t count = 0;
        size_t i = 0;

        while (i + RX_PACKET_LEN <= residual_.size()) {
            if (residual_[i] == RX_HEADER &&
                residual_[i + RX_PACKET_LEN - 1] == RX_TAIL) {
                std::array<uint8_t, RX_PACKET_LEN> pkt;
                std::memcpy(pkt.data(), residual_.data() + i, RX_PACKET_LEN);
                out.push_back(pkt);
                i += RX_PACKET_LEN;
                ++count;
            } else {
                ++i;
            }
        }

        // Erase all scanned bytes (both junk and extracted packets).
        // Only trailing bytes (< RX_PACKET_LEN) that may be a partial packet remain.
        if (i > 0) {
            residual_.erase(residual_.begin(),
                           residual_.begin() + static_cast<ptrdiff_t>(i));
        }
        return count;
    }

    void clear() { residual_.clear(); }

private:
    static constexpr size_t MAX_RESIDUAL = 4096;
    std::vector<uint8_t> residual_;
};

} // namespace trlc
