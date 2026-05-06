#!/usr/bin/env python3
"""
Diagnose intermittent USB-to-CAN freezes on the DK1 follower bus.

Observed symptom: while teleoperating, the left-arm USB2CAN adapter LED stops
its normal red blink and either goes dark or sits at constant blue, the arm
freezes, and the bus is dead until the adapter is power-cycled. The fault
follows the *arm* (not the adapter or the host port), which points at a CAN
bus problem on the left-arm chain (cabling, connector, motor electronics).

This script polls every motor on the chain at ~100 Hz with no torque applied,
so the user can manually move the arm through poses while the script logs
per-motor connectivity and (likely) status / temperature fields.

What it watches per motor:
  * Position, velocity, torque (decoded from the standard MIT-mode reply)
  * The "status byte" data[0] of the reply.  Standard DAMIAO/Cheetah-lineage
    motors encode status/error in this byte; we log every change so any
    transition (e.g. into a comm-error / over-temp / over-voltage state)
    leaves a timestamp in the event log.
  * data[6] / data[7] of the reply.  Per the MIT-mode reply convention these
    are MOS-temperature and rotor-temperature in °C.  They are not parsed by
    DM_CAN.py but the bytes are sitting right there — display them and let
    the user verify against motor heating.
  * Per-motor RX count, time since last reply, longest silence seen.
  * Cumulative motion since startup (so you can verify the joint is actually
    being moved by hand).

What it watches globally:
  * Total bytes received on the serial port.
  * "Junk" bytes — bytes received that don't form a valid 16-byte
    0xAA…0x55 frame.  An adapter in error state may emit malformed or
    short responses; counting these is useful.
  * Bus-freeze events: no bytes received from any motor for > N ms
    (default 300 ms).  This is the smoking gun for the LED-blue freeze.
  * Write failures: serial.write() raising — happens if the adapter
    has fully wedged and USB has dropped out.

Recommended procedure:
  1. Power on the robot, run this script pointed at the affected arm.
  2. Move each joint slowly through its full range, one at a time,
     pausing between joints.  The dashboard's "cum_motion" column shows
     per-joint motion so you can verify you're actually exercising the
     joint (and the script is reading it).
  3. If a freeze occurs, inspect the event log: the BUS_FROZEN
     timestamp can be correlated against the joint that was being moved
     at the time.  MOTOR_SILENT events that fire *before* the global
     BUS_FROZEN may finger a single bad motor as the trigger.
  4. After a freeze: data ceases.  Power-cycle the arm and the adapter,
     then continue.

CSV log columns: ISO timestamp, event, motor, detail.

Usage:
  source port_config.env
  python examples/debug_can_connectivity.py --port "$LEFT_FOLLOWER" \\
      --log /tmp/left_arm_can_$(date +%Y%m%d_%H%M%S).csv
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

try:
    import serial
except ImportError:
    sys.exit("pyserial is required: pip install pyserial")


# --- Motor table (mirrors trlc_dk1_control.motor_chain.DK1MotorChain) -----

@dataclass
class MotorDesc:
    name: str
    slave_id: int
    master_id: int
    motor_type: str   # "DM4340" or "DM4310"


MOTORS_DEFAULT: list[MotorDesc] = [
    MotorDesc("joint_1", 0x01, 0x11, "DM4340"),
    MotorDesc("joint_2", 0x02, 0x12, "DM4340"),
    MotorDesc("joint_3", 0x03, 0x13, "DM4340"),
    MotorDesc("joint_4", 0x04, 0x14, "DM4310"),
    MotorDesc("joint_5", 0x05, 0x15, "DM4310"),
    MotorDesc("joint_6", 0x06, 0x16, "DM4310"),
    MotorDesc("gripper", 0x07, 0x17, "DM4310"),
]

# (q_max, dq_max, tau_max) — must match DM_CAN.MotorControl.Limit_Param
LIMITS = {
    "DM4310": (12.5, 30.0, 10.0),   # also covers gripper
    "DM4340": (12.5,  8.0, 28.0),
}


# --- DM-CAN serial framing (matches csrc/dm_protocol.h) -------------------

# 30-byte send-frame template.  Bytes 13..14 = motor id (LE), 21..28 = data8.
SEND_FRAME_TEMPLATE = bytes([
    0x55, 0xAA, 0x1e, 0x03, 0x01, 0x00, 0x00, 0x00,
    0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
])
RX_HEADER = 0xAA
RX_TAIL = 0x55
RX_LEN = 16


def build_send_frame(motor_id: int, data8: bytes) -> bytes:
    buf = bytearray(SEND_FRAME_TEMPLATE)
    buf[13] = motor_id & 0xff
    buf[14] = (motor_id >> 8) & 0xff
    assert len(data8) == 8
    buf[21:29] = data8
    return bytes(buf)


def build_refresh_frame(slave_id: int) -> bytes:
    """0x7FF + 0xCC + slave_id  →  ask motor for current state (no enable required)."""
    data = bytes([slave_id & 0xff, (slave_id >> 8) & 0xff, 0xCC, 0, 0, 0, 0, 0])
    return build_send_frame(0x7FF, data)


def build_disable_frame(slave_id: int) -> bytes:
    data = bytes([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfd])
    return build_send_frame(slave_id, data)


def build_read_param_frame(slave_id: int, rid: int) -> bytes:
    data = bytes([slave_id & 0xff, (slave_id >> 8) & 0xff, 0x33, rid & 0xff, 0, 0, 0, 0])
    return build_send_frame(0x7FF, data)


# --- Decoding -------------------------------------------------------------

def uint_to_float(x: int, vmin: float, vmax: float, bits: int) -> float:
    return x / ((1 << bits) - 1) * (vmax - vmin) + vmin


@dataclass
class StatePacket:
    status_byte: int        # data[0]
    q: float                # rad
    dq: float               # rad/s
    tau: float              # Nm
    t_mos_byte: int         # data[6]  (likely °C)
    t_rotor_byte: int       # data[7]  (likely °C)


def decode_state(packet: bytes, lim: tuple[float, float, float]) -> StatePacket:
    data = packet[7:15]   # 8 bytes
    q_uint  = (data[1] << 8) | data[2]
    dq_uint = (data[3] << 4) | (data[4] >> 4)
    tau_uint = ((data[4] & 0x0f) << 8) | data[5]
    q_max, dq_max, tau_max = lim
    return StatePacket(
        status_byte = data[0],
        q   = uint_to_float(q_uint,   -q_max,   q_max,   16),
        dq  = uint_to_float(dq_uint,  -dq_max,  dq_max,  12),
        tau = uint_to_float(tau_uint, -tau_max, tau_max, 12),
        t_mos_byte   = data[6],
        t_rotor_byte = data[7],
    )


def decode_can_id(packet: bytes) -> int:
    return packet[3] | (packet[4] << 8) | (packet[5] << 16) | (packet[6] << 24)


def is_param_response(packet: bytes) -> bool:
    return packet[7 + 2] in (0x33, 0x55)


# Parser that mirrors DM_CAN.__extract_packets but also reports junk bytes.
class PacketParser:
    MAX_RESIDUAL = 4096

    def __init__(self) -> None:
        self._buf = bytearray()
        self.junk_bytes = 0  # bytes scanned that didn't start a valid frame

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)
        if len(self._buf) > self.MAX_RESIDUAL:
            drop = len(self._buf) - self.MAX_RESIDUAL
            self.junk_bytes += drop
            del self._buf[:drop]

    def extract(self) -> list[bytes]:
        out: list[bytes] = []
        i = 0
        while i + RX_LEN <= len(self._buf):
            if self._buf[i] == RX_HEADER and self._buf[i + RX_LEN - 1] == RX_TAIL:
                out.append(bytes(self._buf[i:i + RX_LEN]))
                i += RX_LEN
            else:
                i += 1
                self.junk_bytes += 1
        if i > 0:
            del self._buf[:i]
        return out


# --- Per-motor stats ------------------------------------------------------

@dataclass
class MotorStats:
    rx_count: int = 0
    last_rx_ns: int = 0
    last_pos: float = 0.0
    last_vel: float = 0.0
    last_tau: float = 0.0
    last_status: int = -1
    last_t_mos: int = -1
    last_t_rotor: int = -1
    max_t_mos: int = 0
    max_t_rotor: int = 0
    longest_silence_ns: int = 0
    silent: bool = False
    cum_motion: float = 0.0
    motion_window: Deque[float] = field(default_factory=lambda: deque(maxlen=20))


# --- Time / logging helpers -----------------------------------------------

def iso_now() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"


# --- Identification (one-shot read of a few read-only registers) ----------

# RIDs from DM_CAN.DM_variable that are useful for identifying & comparing
# the motors at startup.  Names + RID + ("uint" or "float").
ID_PARAMS = [
    ("hw_ver",   13, "uint"),
    ("sw_ver",   14, "uint"),
    ("sub_ver",  36, "uint"),
    ("SN",       15, "uint"),
    ("MST_ID",    7, "uint"),
    ("ESC_ID",    8, "uint"),
    ("CTRL_MODE", 10, "uint"),
    ("UV_Value",  0, "float"),
    ("OV_Value", 29, "float"),
    ("OT_Value",  2, "float"),
    ("OC_Value",  3, "float"),
    ("MAX_SPD",   6, "float"),
]


def _decode_param_value(packet: bytes, kind: str):
    data = packet[7:15]
    raw = bytes(data[4:8])
    if kind == "uint":
        return struct.unpack("<I", raw)[0]
    else:
        return struct.unpack("<f", raw)[0]


def read_motor_identification(ser, motors: list[MotorDesc], parser: PacketParser,
                              log_event) -> dict[int, dict[str, object]]:
    """Best-effort one-shot read of identifying registers per motor."""
    info: dict[int, dict[str, object]] = {m.slave_id: {} for m in motors}

    for m in motors:
        for name, rid, kind in ID_PARAMS:
            try:
                ser.write(build_read_param_frame(m.slave_id, rid))
            except serial.SerialException as e:
                log_event("WRITE_FAIL", motor=m.name, detail=f"id-read {name}: {e}")
                continue
            # Wait briefly for the reply
            deadline = time.monotonic() + 0.05
            value = None
            while time.monotonic() < deadline:
                n = ser.in_waiting
                if n:
                    parser.feed(ser.read(n))
                for pkt in parser.extract():
                    if pkt[1] != 0x11 or not is_param_response(pkt):
                        continue
                    data = pkt[7:15]
                    # data[0..1] = slave_id (LE), data[3] = rid
                    slave_in_pkt = (data[1] << 8) | data[0]
                    if slave_in_pkt == m.slave_id and data[3] == rid:
                        value = _decode_param_value(pkt, kind)
                        break
                if value is not None:
                    break
                time.sleep(0.001)
            info[m.slave_id][name] = value
    return info


# --- Main loop ------------------------------------------------------------

# MIT-mode reply byte 0 layout (DAMIAO / Mini-Cheetah lineage):
#   low nibble  = motor slave id (the parser already uses this for routing)
#   high nibble = status / error code
# The codes below are the standard Cheetah-lineage values.  Not authoritative
# for DAMIAO firmware specifically; the script also logs every change of
# data[0] so anything unexpected leaves a timestamp regardless.
STATUS_NIBBLE_HINT = {
    0x0: "DISABLED",
    0x1: "ENABLED",
    0x8: "OVERVOLT?",
    0x9: "UNDERVOLT?",
    0xA: "OVERCURR?",
    0xB: "MOS_OVERTEMP?",
    0xC: "ROTOR_OVERTEMP?",
    0xD: "COMM_LOST?",
    0xE: "OVERLOAD?",
}


def status_label(byte_value: int) -> str:
    if byte_value < 0:
        return "—"
    high = (byte_value >> 4) & 0x0f   # status nibble (low nibble = slave id)
    hint = STATUS_NIBBLE_HINT.get(high, "?")
    return f"0x{byte_value:02X} {hint}"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--port", required=True,
                   help="serial device, e.g. /dev/serial/by-path/...follower")
    p.add_argument("--baudrate", type=int, default=921600)
    p.add_argument("--rate-hz", type=float, default=100.0,
                   help="poll cycle rate; one refresh per motor per cycle (default 100 Hz)")
    p.add_argument("--motors", default="all",
                   help="comma-separated motor names (default: all 7)")
    p.add_argument("--silence-ms", type=float, default=200.0,
                   help="mark a single motor SILENT after this much time without an RX (default 200 ms)")
    p.add_argument("--bus-freeze-ms", type=float, default=300.0,
                   help="mark BUS FROZEN when *no* motor has replied for this long (default 300 ms)")
    p.add_argument("--dashboard-s", type=float, default=1.0,
                   help="dashboard reprint period (default 1.0 s)")
    p.add_argument("--log", default=None,
                   help="optional CSV event log path")
    p.add_argument("--no-disable", action="store_true",
                   help="skip the initial torque-off disable broadcast (NOT recommended)")
    p.add_argument("--no-id-read", action="store_true",
                   help="skip the one-shot identification register read")
    args = p.parse_args()

    if args.motors == "all":
        motors = list(MOTORS_DEFAULT)
    else:
        wanted = {x.strip() for x in args.motors.split(",")}
        motors = [m for m in MOTORS_DEFAULT if m.name in wanted]
        if not motors:
            sys.exit(f"No motors selected from: {args.motors}")

    # --- Open port -------------------------------------------------------
    print(f"Opening {args.port} @ {args.baudrate} baud", flush=True)
    ser = serial.Serial(args.port, args.baudrate, timeout=0)
    time.sleep(0.5)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    parser = PacketParser()
    stats: dict[int, MotorStats] = {m.slave_id: MotorStats() for m in motors}

    log_fh = None
    if args.log:
        log_fh = open(args.log, "w", buffering=1)
        log_fh.write("timestamp_iso,event,motor,detail\n")

    def log_event(event: str, motor: str = "", detail: str = "") -> None:
        ts = iso_now()
        msg = f"{ts}  [{event:<16}] {motor:<10} {detail}".rstrip()
        print(msg, flush=True)
        if log_fh:
            # CSV-escape commas in detail
            d = detail.replace(",", ";")
            log_fh.write(f"{ts},{event},{motor},{d}\n")

    log_event("START", detail=f"port={args.port} motors={[m.name for m in motors]}")

    # --- Disable everything so the arm can be moved by hand --------------
    if not args.no_disable:
        log_event("DISABLE_ALL", detail="sending disable to every motor")
        for m in motors:
            try:
                ser.write(build_disable_frame(m.slave_id))
            except serial.SerialException as e:
                log_event("WRITE_FAIL", motor=m.name, detail=f"disable: {e}")
            time.sleep(0.005)
        time.sleep(0.2)
        ser.reset_input_buffer()
        parser = PacketParser()  # drop any disable acks

    # --- One-shot identification register dump ---------------------------
    if not args.no_id_read:
        log_event("ID_READ", detail="reading hw/sw/SN/threshold registers")
        info = read_motor_identification(ser, motors, parser, log_event)
        ser.reset_input_buffer()
        parser = PacketParser()
        for m in motors:
            kv = info.get(m.slave_id, {})
            parts = []
            for name, _, _ in ID_PARAMS:
                v = kv.get(name)
                if v is None:
                    parts.append(f"{name}=?")
                elif isinstance(v, float):
                    parts.append(f"{name}={v:.3f}")
                else:
                    parts.append(f"{name}={v}")
            log_event("ID", motor=m.name, detail=" ".join(parts))

    # --- Polling loop ----------------------------------------------------
    period = 1.0 / args.rate_hz
    silence_threshold_ns = int(args.silence_ms * 1e6)
    bus_freeze_threshold_ns = int(args.bus_freeze_ms * 1e6)

    last_global_rx_ns = time.monotonic_ns()
    bus_frozen = False
    last_dashboard_t = time.monotonic()
    next_tick = time.monotonic() + period
    total_rx_bytes = 0
    last_dash_motion: dict[int, float] = {m.slave_id: 0.0 for m in motors}

    print()
    log_event("RUN", detail=f"polling at {args.rate_hz:.0f} Hz "
                            f"silence>{args.silence_ms:.0f}ms freeze>{args.bus_freeze_ms:.0f}ms")
    print(flush=True)

    try:
        while True:
            now = time.monotonic()
            now_ns = time.monotonic_ns()

            # 1. Drain RX buffer (pipeline: replies from previous cycle).
            avail = ser.in_waiting
            if avail:
                chunk = ser.read(avail)
                if chunk:
                    parser.feed(chunk)
                    total_rx_bytes += len(chunk)
                    last_global_rx_ns = now_ns

            # 2. Parse and update per-motor state.
            for pkt in parser.extract():
                if pkt[1] != 0x11:
                    continue
                if is_param_response(pkt):
                    continue
                can_id = decode_can_id(pkt)
                matched: MotorDesc | None = None
                for m in motors:
                    if can_id == m.slave_id or can_id == m.master_id:
                        matched = m
                        break
                    if can_id == 0 and (pkt[7] & 0x0f) == (m.master_id & 0x0f):
                        matched = m
                        break
                if matched is None:
                    continue

                lim = LIMITS[matched.motor_type]
                st = decode_state(pkt, lim)
                ms = stats[matched.slave_id]

                # Per-motor recovery
                if ms.silent:
                    silence_ms = (now_ns - ms.last_rx_ns) / 1e6 if ms.last_rx_ns else -1
                    log_event("MOTOR_RECOVERED", motor=matched.name,
                              detail=f"after {silence_ms:.1f}ms longest_seen={ms.longest_silence_ns/1e6:.1f}ms")
                    ms.silent = False

                # Status nibble change → log it.  The low nibble of data[0]
                # is the slave id (not status), so only the high nibble matters.
                new_status_high = (st.status_byte >> 4) & 0x0f
                old_status_high = ((ms.last_status >> 4) & 0x0f) if ms.last_status != -1 else -1
                if new_status_high != old_status_high and ms.last_status != -1:
                    log_event("STATUS_CHANGE", motor=matched.name,
                              detail=f"data[0] {ms.last_status:#04x} -> {st.status_byte:#04x} "
                                     f"(status_nibble {old_status_high:x} -> {new_status_high:x})")
                ms.last_status = st.status_byte

                # Temperature byte changes (don't spam — only on change of >=2°)
                if abs(st.t_mos_byte - ms.last_t_mos) >= 2 and ms.last_t_mos != -1:
                    log_event("T_MOS", motor=matched.name,
                              detail=f"{ms.last_t_mos} -> {st.t_mos_byte} (data[6])")
                if abs(st.t_rotor_byte - ms.last_t_rotor) >= 2 and ms.last_t_rotor != -1:
                    log_event("T_ROTOR", motor=matched.name,
                              detail=f"{ms.last_t_rotor} -> {st.t_rotor_byte} (data[7])")
                ms.last_t_mos = st.t_mos_byte
                ms.last_t_rotor = st.t_rotor_byte
                ms.max_t_mos = max(ms.max_t_mos, st.t_mos_byte)
                ms.max_t_rotor = max(ms.max_t_rotor, st.t_rotor_byte)

                # Motion accumulation (cumulative absolute angular travel)
                if ms.rx_count > 0:
                    delta = abs(st.q - ms.last_pos)
                    if delta < 1.0:        # ignore wrap / glitch jumps
                        ms.cum_motion += delta
                        ms.motion_window.append(delta)

                ms.last_pos = st.q
                ms.last_vel = st.dq
                ms.last_tau = st.tau
                ms.last_rx_ns = now_ns
                ms.rx_count += 1

            # 3. Per-motor silence detection.
            for m in motors:
                ms = stats[m.slave_id]
                if ms.last_rx_ns == 0:
                    continue
                silence_ns = now_ns - ms.last_rx_ns
                if silence_ns > ms.longest_silence_ns:
                    ms.longest_silence_ns = silence_ns
                if silence_ns > silence_threshold_ns and not ms.silent:
                    ms.silent = True
                    log_event("MOTOR_SILENT", motor=m.name,
                              detail=f"{silence_ns/1e6:.1f}ms since last reply "
                                     f"(last_pos={ms.last_pos:+.3f} last_status={ms.last_status:#04x})")

            # 4. Global bus-freeze detection (correlates with the blue/dark LED).
            global_silence_ns = now_ns - last_global_rx_ns
            if not bus_frozen and global_silence_ns > bus_freeze_threshold_ns:
                bus_frozen = True
                log_event("BUS_FROZEN",
                          detail=f"no bytes from any motor for {global_silence_ns/1e6:.1f}ms — "
                                 f"USB-CAN adapter likely hung (LED blue / off)")
            elif bus_frozen and global_silence_ns < bus_freeze_threshold_ns / 3:
                bus_frozen = False
                log_event("BUS_RECOVERED",
                          detail=f"RX resumed after blackout")

            # 5. Send refresh frames for every motor.
            for m in motors:
                try:
                    ser.write(build_refresh_frame(m.slave_id))
                except serial.SerialException as e:
                    log_event("WRITE_FAIL", motor=m.name, detail=f"refresh: {e}")

            # 6. Dashboard.
            if now - last_dashboard_t >= args.dashboard_s:
                last_dashboard_t = now
                gms = global_silence_ns / 1e6
                header = (
                    f"\n=== {time.strftime('%H:%M:%S')}  "
                    f"global_last_rx={gms:7.1f}ms  "
                    f"bus_frozen={bus_frozen}  "
                    f"total_rx={total_rx_bytes}B  "
                    f"junk={parser.junk_bytes}B"
                    f" ==="
                )
                cols = (
                    f"  {'motor':<8s} {'slave':>5s} "
                    f"{'pos':>9s} {'vel':>7s} {'tau':>7s} "
                    f"{'rx':>8s} {'last_ms':>8s} {'maxsil_ms':>10s} "
                    f"{'cum_mot':>8s} {'window':>8s} "
                    f"{'T_MOS':>5s} {'T_ROT':>5s}  status"
                )
                lines = [header, cols]
                for m in motors:
                    ms = stats[m.slave_id]
                    if ms.last_rx_ns == 0:
                        last_ms = float("inf")
                    else:
                        last_ms = (now_ns - ms.last_rx_ns) / 1e6
                    win = sum(ms.motion_window)
                    lines.append(
                        f"  {m.name:<8s} 0x{m.slave_id:02X}   "
                        f"{ms.last_pos:+9.3f} {ms.last_vel:+7.3f} {ms.last_tau:+7.3f} "
                        f"{ms.rx_count:>8d} {last_ms:>8.1f} {ms.longest_silence_ns/1e6:>10.1f} "
                        f"{ms.cum_motion:>8.3f} {win:>8.3f} "
                        f"{ms.last_t_mos:>5d} {ms.last_t_rotor:>5d}  {status_label(ms.last_status)}"
                        f"{'  SILENT' if ms.silent else ''}"
                    )
                print("\n".join(lines), flush=True)

            # 7. Pace the cycle.
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif sleep_for < -0.5:
                next_tick = time.monotonic()  # fell badly behind, resync
            next_tick += period

    except KeyboardInterrupt:
        log_event("STOP", detail="user interrupt (Ctrl-C)")
    finally:
        # Best-effort: leave motors disabled.
        try:
            for m in motors:
                ser.write(build_disable_frame(m.slave_id))
                time.sleep(0.005)
        except Exception:
            pass
        if log_fh:
            log_fh.close()
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
