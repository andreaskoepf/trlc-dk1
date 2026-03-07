#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>

namespace trlc {

struct SafetyState {
    bool damping_mode = false;
    int overcurrent_count = 0;
    int overspeed_count = 0;
};

// Apply safety checks, modifying q_des, kp, and tau_ff in-place.
// ndof: number of joints (typically 6)
inline void apply_safety(
    const double* pos, const double* vel, const double* torque,
    double* q_des, double* kp, double* tau_ff,
    const double* pos_limits_lo, const double* pos_limits_hi,
    const double* torque_limits, const double* velocity_limits,
    double limit_buffer,
    int overcurrent_threshold, int overspeed_threshold,
    int ndof,
    SafetyState& state)
{
    // Position clamping with buffer
    for (int i = 0; i < ndof; ++i) {
        double lo = pos_limits_lo[i] + limit_buffer;
        double hi = pos_limits_hi[i] - limit_buffer;
        q_des[i] = std::clamp(q_des[i], lo, hi);
    }

    // Torque limit clipping
    for (int i = 0; i < ndof; ++i) {
        tau_ff[i] = std::clamp(tau_ff[i], -torque_limits[i], torque_limits[i]);
    }

    // Over-speed detection
    bool any_overspeed = false;
    for (int i = 0; i < ndof; ++i) {
        if (std::abs(vel[i]) > velocity_limits[i]) {
            any_overspeed = true;
            break;
        }
    }

    if (any_overspeed) {
        ++state.overspeed_count;
        if (state.overspeed_count >= overspeed_threshold) {
            state.damping_mode = true;
        }
    } else {
        state.overspeed_count = std::max(0, state.overspeed_count - 1);
    }

    // Over-current detection
    bool any_over = false;
    for (int i = 0; i < ndof; ++i) {
        if (std::abs(torque[i]) > torque_limits[i]) {
            any_over = true;
            break;
        }
    }

    if (any_over) {
        ++state.overcurrent_count;
        if (state.overcurrent_count >= overcurrent_threshold) {
            state.damping_mode = true;
        }
    } else {
        state.overcurrent_count = std::max(0, state.overcurrent_count - 1);
    }

    // In damping mode: zero stiffness and feedforward
    if (state.damping_mode) {
        for (int i = 0; i < ndof; ++i) {
            kp[i] = 0.0;
            tau_ff[i] = 0.0;
            q_des[i] = pos[i];  // hold current position for when we exit damping
        }
    }
}

} // namespace trlc
