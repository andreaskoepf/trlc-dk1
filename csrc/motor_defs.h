#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace trlc {

enum class MotorType : int {
    DM4310     = 0,
    DM4310_48V = 1,
    DM4340     = 2,
    DM4340_48V = 3,
    DM6006     = 4,
    DM8006     = 5,
    DM8009     = 6,
    DM10010L   = 7,
    DM10010    = 8,
    DMH3510    = 9,
    DMH6215    = 10,
    DMG6220    = 11,
};

struct MotorLimits {
    float q_max;
    float dq_max;
    float tau_max;
};

// Matches DM_CAN.py Limit_Param exactly
inline constexpr std::array<MotorLimits, 12> MOTOR_LIMITS = {{
    {12.5f, 30.0f,  10.0f},   // DM4310
    {12.5f, 50.0f,  10.0f},   // DM4310_48V
    {12.5f,  8.0f,  28.0f},   // DM4340
    {12.5f, 10.0f,  28.0f},   // DM4340_48V
    {12.5f, 45.0f,  20.0f},   // DM6006
    {12.5f, 45.0f,  40.0f},   // DM8006
    {12.5f, 45.0f,  54.0f},   // DM8009
    {12.5f, 25.0f, 200.0f},   // DM10010L
    {12.5f, 20.0f, 200.0f},   // DM10010
    {12.5f, 280.0f,  1.0f},   // DMH3510
    {12.5f, 45.0f,  10.0f},   // DMH6215
    {12.5f, 45.0f,  10.0f},   // DMG6220
}};

inline const MotorLimits& get_limits(MotorType type) {
    int idx = static_cast<int>(type);
    if (idx < 0 || idx >= static_cast<int>(MOTOR_LIMITS.size())) {
        throw std::out_of_range("Invalid MotorType index: " + std::to_string(idx));
    }
    return MOTOR_LIMITS[static_cast<size_t>(idx)];
}

struct MotorDescriptor {
    std::string name;
    MotorType type;
    uint16_t slave_id;
    uint16_t master_id;
};

} // namespace trlc
