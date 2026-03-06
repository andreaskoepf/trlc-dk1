#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>

#include "dm_protocol.h"
#include "motor_defs.h"

using namespace trlc;

static void test_float_to_uint_roundtrip() {
    // Test quantization round-trip for various ranges
    struct TestCase { float val; float min; float max; int bits; };
    TestCase cases[] = {
        {0.0f, -12.5f, 12.5f, 16},
        {5.0f, -12.5f, 12.5f, 16},
        {-5.0f, -12.5f, 12.5f, 16},
        {12.5f, -12.5f, 12.5f, 16},
        {-12.5f, -12.5f, 12.5f, 16},
        {0.0f, -30.0f, 30.0f, 12},
        {15.0f, -30.0f, 30.0f, 12},
        {250.0f, 0.0f, 500.0f, 12},
        {2.5f, 0.0f, 5.0f, 12},
    };

    for (const auto& tc : cases) {
        uint16_t u = float_to_uint(tc.val, tc.min, tc.max, tc.bits);
        float back = uint_to_float(u, tc.min, tc.max, tc.bits);
        float span = tc.max - tc.min;
        float tolerance = span / ((1 << tc.bits) - 1) * 1.5f;
        assert(std::abs(back - tc.val) <= tolerance);
    }
    std::printf("  float_to_uint roundtrip: PASS\n");
}

static void test_float_to_uint_clamp() {
    // Values outside range should be clamped
    uint16_t u = float_to_uint(100.0f, -12.5f, 12.5f, 16);
    float back = uint_to_float(u, -12.5f, 12.5f, 16);
    assert(std::abs(back - 12.5f) < 0.01f);

    u = float_to_uint(-100.0f, -12.5f, 12.5f, 16);
    back = uint_to_float(u, -12.5f, 12.5f, 16);
    assert(std::abs(back - (-12.5f)) < 0.01f);

    std::printf("  float_to_uint clamping: PASS\n");
}

static void test_mit_frame_format() {
    // Build a MIT frame and verify the structure
    uint8_t frame[30];
    MotorLimits lim = {12.5f, 8.0f, 28.0f};  // DM4340

    build_mit_frame(frame, 0x01, 80.0f, 3.0f, 1.0f, 0.0f, 5.0f, lim);

    // Check header bytes
    assert(frame[0] == 0x55);
    assert(frame[1] == 0xAA);
    assert(frame[2] == 0x1E);  // 30 = frame length

    // Check motor ID
    assert(frame[13] == 0x01);
    assert(frame[14] == 0x00);

    // Data length marker
    assert(frame[18] == 0x08);

    // Verify we can decode back (by looking at the data bytes at offset 21)
    const uint8_t* data = frame + 21;

    // Reconstruct q
    uint16_t q_uint = (static_cast<uint16_t>(data[0]) << 8) | data[1];
    float q_back = uint_to_float(q_uint, -12.5f, 12.5f, 16);
    assert(std::abs(q_back - 1.0f) < 0.01f);

    // Reconstruct dq
    uint16_t dq_uint = (static_cast<uint16_t>(data[2]) << 4) | (data[3] >> 4);
    float dq_back = uint_to_float(dq_uint, -8.0f, 8.0f, 12);
    assert(std::abs(dq_back - 0.0f) < 0.05f);

    // Reconstruct kp
    uint16_t kp_uint = ((data[3] & 0x0F) << 8) | data[4];
    float kp_back = uint_to_float(kp_uint, 0.0f, 500.0f, 12);
    assert(std::abs(kp_back - 80.0f) < 1.0f);

    // Reconstruct kd
    uint16_t kd_uint = (static_cast<uint16_t>(data[5]) << 4) | (data[6] >> 4);
    float kd_back = uint_to_float(kd_uint, 0.0f, 5.0f, 12);
    assert(std::abs(kd_back - 3.0f) < 0.01f);

    // Reconstruct tau
    uint16_t tau_uint = ((data[6] & 0x0F) << 8) | data[7];
    float tau_back = uint_to_float(tau_uint, -28.0f, 28.0f, 12);
    assert(std::abs(tau_back - 5.0f) < 0.1f);

    std::printf("  MIT frame format: PASS\n");
}

static void test_emit_frame_format() {
    uint8_t frame[30];
    build_emit_frame(frame, 0x07, -2.35f, 3000.0f, 1058.0f);

    // Motor ID should be 0x307
    assert(frame[13] == 0x07);
    assert(frame[14] == 0x03);

    // Position is IEEE 754 float at offset 21
    float pos;
    std::memcpy(&pos, frame + 21, 4);
    assert(std::abs(pos - (-2.35f)) < 0.001f);

    // Velocity at offset 25 (little-endian uint16)
    uint16_t vel = frame[25] | (static_cast<uint16_t>(frame[26]) << 8);
    assert(vel == 3000);

    // Current at offset 27 (little-endian uint16)
    uint16_t ides = frame[27] | (static_cast<uint16_t>(frame[28]) << 8);
    assert(ides == 1058);

    std::printf("  EMIT frame format: PASS\n");
}

static void test_enable_disable_frames() {
    uint8_t frame[30];

    build_enable_frame(frame, 0x05);
    assert(frame[13] == 0x05);
    assert(frame[28] == 0xFC);  // enable command byte in data[7]

    build_disable_frame(frame, 0x05);
    assert(frame[28] == 0xFD);

    build_set_zero_frame(frame, 0x05);
    assert(frame[28] == 0xFE);

    std::printf("  enable/disable/set-zero frames: PASS\n");
}

static void test_refresh_frame() {
    uint8_t frame[30];
    build_refresh_frame(frame, 0x03);

    // CAN ID should be 0x7FF
    assert(frame[13] == 0xFF);
    assert(frame[14] == 0x07);

    // Data: slave_id_lo, slave_id_hi, 0xCC, ...
    assert(frame[21] == 0x03);
    assert(frame[22] == 0x00);
    assert(frame[23] == 0xCC);

    std::printf("  refresh frame: PASS\n");
}

static void test_switch_mode_frame() {
    uint8_t frame[30];
    build_switch_mode_frame(frame, 0x04, 1);  // MIT mode

    assert(frame[13] == 0xFF);
    assert(frame[14] == 0x07);
    assert(frame[21] == 0x04);  // slave_id
    assert(frame[23] == 0x55);  // write command
    assert(frame[24] == 10);    // RID = CTRL_MODE
    assert(frame[25] == 1);     // mode value

    std::printf("  switch mode frame: PASS\n");
}

static void test_packet_parser() {
    PacketParser parser;

    // Build a fake 16-byte receive packet
    uint8_t pkt[16] = {};
    pkt[0] = 0xAA;   // header
    pkt[15] = 0x55;  // tail
    pkt[1] = 0x11;   // CMD
    pkt[3] = 0x01;   // CAN ID low byte (slave_id=1)

    // Set motor state data at bytes 7-12
    // q = 0 -> q_uint = 32768 (midpoint for 16-bit with [-12.5, 12.5])
    uint16_t q_uint = 32768;  // approximately 0.0
    pkt[8] = (q_uint >> 8) & 0xFF;
    pkt[9] = q_uint & 0xFF;
    // dq and tau at midpoint too
    uint16_t dq_uint = 2048;
    pkt[10] = dq_uint >> 4;
    pkt[11] = (dq_uint & 0x0F) << 4;
    uint16_t tau_uint = 2048;
    pkt[11] |= (tau_uint >> 8) & 0x0F;
    pkt[12] = tau_uint & 0xFF;

    // Feed packet with some garbage before it
    uint8_t garbage[] = {0x12, 0x34, 0x56};
    parser.feed(garbage, 3);
    parser.feed(pkt, 16);

    // Also add a second packet
    pkt[3] = 0x02;  // different CAN ID
    parser.feed(pkt, 16);

    std::vector<std::array<uint8_t, 16>> packets;
    size_t count = parser.extract(packets);
    assert(count == 2);
    assert(packets[0][3] == 0x01);
    assert(packets[1][3] == 0x02);

    // Decode state from first packet
    MotorLimits lim = {12.5f, 30.0f, 10.0f};  // DM4310
    auto state = decode_motor_state(packets[0].data(), lim);
    assert(std::abs(state.q) < 0.01f);  // should be close to 0

    std::printf("  packet parser: PASS\n");
}

static void test_partial_packet() {
    PacketParser parser;

    uint8_t pkt[16] = {};
    pkt[0] = 0xAA;
    pkt[15] = 0x55;
    pkt[1] = 0x11;

    // Feed first half
    parser.feed(pkt, 8);
    std::vector<std::array<uint8_t, 16>> packets;
    size_t count = parser.extract(packets);
    assert(count == 0);

    // Feed second half
    parser.feed(pkt + 8, 8);
    count = parser.extract(packets);
    assert(count == 1);

    std::printf("  partial packet reassembly: PASS\n");
}

int main() {
    std::printf("dm_protocol tests:\n");
    test_float_to_uint_roundtrip();
    test_float_to_uint_clamp();
    test_mit_frame_format();
    test_emit_frame_format();
    test_enable_disable_frames();
    test_refresh_frame();
    test_switch_mode_frame();
    test_packet_parser();
    test_partial_packet();
    std::printf("All dm_protocol tests passed!\n");
    return 0;
}
