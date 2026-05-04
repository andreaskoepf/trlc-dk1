from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_URDF = str(_REPO_ROOT / "urdf" / "follower" / "TRLC-DK1-Follower.urdf")


DM4310_IDX = 0   # Limit_Param[0] = [12.5, 30, 10]
DM4340_IDX = 2   # Limit_Param[2] = [12.5, 8, 28]

DM4310_Q_MAX = 12.5    # rad
DM4310_DQ_MAX = 30.0   # rad/s
DM4310_T_MAX = 10.0    # Nm

DM4340_Q_MAX = 12.5    # rad
DM4340_DQ_MAX = 8.0    # rad/s  (52.5 rpm)
DM4340_T_MAX = 28.0    # Nm

# MIT gain ranges (from DM_CAN.py float_to_uint constraints)
KP_MAX = 500.0
KD_MAX = 5.0


@dataclass
class DK1RobotConfig:
    """All tunable parameters for the TRLC-DK1 control stack."""

    # Serial communication
    serial_port: str = "/dev/ttyACM0"
    serial_timeout: float = 0.005   # 5 ms — must be short for 250 Hz loop

    # Thread rates
    motor_thread_hz: float = 250.0
    server_thread_hz: float = 300.0

    # MIT PD gains for 6 arm joints [j1, j2, j3, j4, j5, j6]
    arm_kp: np.ndarray = field(
        default_factory=lambda: np.array([80.0, 70.0, 60.0, 20.0, 20.0, 10.0])
    )
    arm_kd: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 4.0, 1.0, 1.0, 1.0])
    )

    # Joint position limits (radians), shape (6, 2) — [min, max] per joint
    # Joints 1-3 (DM4340): physically limited by arm geometry; use conservative ±π
    # Joints 4-5 (DM4310): taken from follower.py JOINT_LIMITS
    # Joint 6   (DM4310): full ±π
    joint_pos_limits: np.ndarray = field(
        default_factory=lambda: np.array([
            [-math.pi,       math.pi      ],   # joint_1
            [-math.pi,       math.pi      ],   # joint_2
            [-math.pi,       math.pi      ],   # joint_3
            [-100*math.pi/180, 100*math.pi/180],  # joint_4
            [-90*math.pi/180,  90*math.pi/180 ],  # joint_5
            [-math.pi,       math.pi      ],   # joint_6
        ])
    )

    # Joint torque limits (Nm) per joint — matches motor T_MAX
    joint_torque_limits: np.ndarray = field(
        default_factory=lambda: np.array([28.0, 28.0, 28.0, 10.0, 10.0, 10.0])
    )

    # URDF path (used for kinematics / visualisation)
    urdf_path: str = _DEFAULT_URDF

    # Gravity compensation
    mjcf_path: str = _DEFAULT_URDF   # path to MuJoCo XML; empty = gravity comp disabled
    gravity_comp_scale: float = 1.0  # tune empirically

    # Joint velocity limits (rad/s) per joint — operational safety limit
    joint_velocity_limits: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 5.0, 15.0, 15.0, 15.0])
    )

    # Slew rate limit: max position change per cycle (rad/cycle) per joint
    # At 250 Hz: 0.02 rad/cycle = 5 rad/s, 0.06 rad/cycle = 15 rad/s. 0.0 disables.
    max_pos_delta_per_cycle: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.06, 0.06, 0.06])
    )

    # Safety watchdog. When no fresh command arrives for this many
    # seconds the C++ loop re-targets ``cmd.q_des = cur_pos`` (HOLD)
    # and then the slew limiter reverts to the safe per-cycle behaviour
    # of "motor keeps its last good MIT frame indefinitely".
    # 0.5 s was misleading — it fires on routine sub-second NVENC stalls
    # and turns a recoverable freeze into a visible drop. Bumped to
    # 3.0 s so the watchdog still catches a genuinely stuck Python but
    # stays out of the way during normal-cause stalls (mp4 av.open, gc,
    # etc.).
    command_timeout_s: float = 3.0

    overcurrent_threshold: int = 20   # consecutive over-limit torque counts before damping
    overspeed_threshold: int = 5      # consecutive over-limit velocity counts before damping
    min_motors_required: int = 6      # throw if fewer arm motors respond during init
    gripper_cal_timeout_s: float = 10.0  # wall-clock timeout for gripper calibration

    # Communication loss detection
    max_consecutive_empty_cycles: int = 50  # cycles with 0 bytes before comm loss (200ms at 250Hz)

    # Shutdown
    disable_torque_on_disconnect: bool = True

    # Gripper parameters
    gripper_open_pos: float = 0.0     # rad offset from hard stop after calibration
    gripper_closed_pos: float = -4.7  # rad
    max_gripper_torque_nm: float = 1.0
    DM4310_TORQUE_CONSTANT: float = 0.945  # Nm/A
    EMIT_VELOCITY_SCALE: float = 100.0     # rad/s multiplier for EMIT mode
    EMIT_CURRENT_SCALE: float = 1000.0     # A multiplier for EMIT mode


def DK1_DEFAULT_CONFIG(serial_port: str, mjcf_path: str = _DEFAULT_URDF) -> DK1RobotConfig:
    """Return a default DK1RobotConfig for the standard 6-DOF arm + gripper."""
    return DK1RobotConfig(
        serial_port=serial_port,
        mjcf_path=mjcf_path,
    )
