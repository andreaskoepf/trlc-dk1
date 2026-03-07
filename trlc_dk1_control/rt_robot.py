"""
DK1RobotRT — Drop-in replacement for DK1Robot using the C++ RT control loop.

Uses the _trlc_dk1_rt nanobind extension for a single-threaded 250 Hz
control loop with optional PREEMPT_RT support and lock-free communication.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import DK1RobotConfig

logger = logging.getLogger(__name__)

# Default motor configuration matching motor_chain.py
_DEFAULT_MOTORS = [
    {"name": "joint_1", "type": "DM4340",  "slave_id": 0x01, "master_id": 0x11},
    {"name": "joint_2", "type": "DM4340",  "slave_id": 0x02, "master_id": 0x12},
    {"name": "joint_3", "type": "DM4340",  "slave_id": 0x03, "master_id": 0x13},
    {"name": "joint_4", "type": "DM4310",  "slave_id": 0x04, "master_id": 0x14},
    {"name": "joint_5", "type": "DM4310",  "slave_id": 0x05, "master_id": 0x15},
    {"name": "joint_6", "type": "DM4310",  "slave_id": 0x06, "master_id": 0x16},
    {"name": "gripper",  "type": "DM4310",  "slave_id": 0x07, "master_id": 0x17},
]


class DK1RobotRT:
    """
    Drop-in replacement for DK1Robot using the C++ RT control loop.

    Provides the same public API as DK1Robot for use with DK1Follower
    and other orchestration code.
    """

    def __init__(self, config: DK1RobotConfig) -> None:
        try:
            from trlc_dk1_control._trlc_dk1_rt import RtControlLoop, RtLoopConfig, MotorDescriptor, MotorType
        except ImportError as e:
            raise ImportError(
                "C++ RT extension not available. Install with: "
                "pip install -e '.[rt]'"
            ) from e

        self._config = config
        self._RtControlLoop = RtControlLoop
        self._loop = None

        # Build RtLoopConfig from DK1RobotConfig
        rt_cfg = RtLoopConfig()
        rt_cfg.serial_port = config.serial_port
        rt_cfg.loop_hz = config.motor_thread_hz

        # Motor descriptors
        motor_type_map = {
            "DM4310": MotorType.DM4310,
            "DM4310_48V": MotorType.DM4310_48V,
            "DM4340": MotorType.DM4340,
            "DM4340_48V": MotorType.DM4340_48V,
        }

        motors = []
        for m in _DEFAULT_MOTORS:
            desc = MotorDescriptor()
            desc.name = m["name"]
            desc.type = motor_type_map[m["type"]]
            desc.slave_id = m["slave_id"]
            desc.master_id = m["master_id"]
            motors.append(desc)
        rt_cfg.motors = motors

        # Gains
        rt_cfg.default_kp = np.asarray(config.arm_kp, dtype=np.float64)
        rt_cfg.default_kd = np.asarray(config.arm_kd, dtype=np.float64)

        # Joint limits (flatten from (6,2) to (12,) interleaved [lo0,hi0,...])
        limits = np.asarray(config.joint_pos_limits, dtype=np.float64)
        rt_cfg.joint_pos_limits = limits.flatten()

        rt_cfg.joint_torque_limits = np.asarray(config.joint_torque_limits, dtype=np.float64)
        rt_cfg.limit_buffer = 0.05

        # Gravity compensation
        if config.mjcf_path:
            rt_cfg.model_path = str(Path(config.mjcf_path).resolve())
        rt_cfg.gravity_comp_scale = config.gravity_comp_scale

        # Safety
        rt_cfg.command_timeout_s = config.command_timeout_s
        rt_cfg.overcurrent_threshold = config.overcurrent_threshold

        # Gripper
        rt_cfg.gripper_open_pos = config.gripper_open_pos
        rt_cfg.gripper_closed_pos = config.gripper_closed_pos
        rt_cfg.max_gripper_torque_nm = config.max_gripper_torque_nm
        rt_cfg.torque_constant = config.DM4310_TORQUE_CONSTANT
        rt_cfg.emit_velocity_scale = config.EMIT_VELOCITY_SCALE
        rt_cfg.emit_current_scale = config.EMIT_CURRENT_SCALE
        rt_cfg.disable_torque_on_disconnect = config.disable_torque_on_disconnect

        self._rt_cfg = rt_cfg

    def connect(self) -> None:
        """Start the C++ RT control loop."""
        self._loop = self._RtControlLoop(self._rt_cfg)
        self._loop.start()
        logger.info(
            "DK1RobotRT connected (RT active: %s)", self._loop.is_rt_active()
        )

    def disconnect(self) -> None:
        """Stop the C++ RT control loop."""
        if self._loop is not None:
            self._loop.stop()
            self._loop = None
        logger.info("DK1RobotRT disconnected")

    def command_joint_pos(self, q_des: np.ndarray) -> None:
        """Set target joint positions for the 6 arm joints (radians)."""
        if q_des.shape != (6,):
            raise ValueError(f"Expected shape (6,), got {q_des.shape}")
        if self._loop is not None:
            self._loop.command_joint_pos(np.asarray(q_des, dtype=np.float64))

    def command_gripper(self, normalized_pos: float) -> None:
        """Set gripper target (0.0=open, 1.0=closed)."""
        if self._loop is not None:
            self._loop.command_gripper(float(normalized_pos))

    def get_joint_state(self) -> dict[str, np.ndarray]:
        """Return arm joint state as dict with 'pos', 'vel', 'torque' arrays."""
        if self._loop is None:
            return {
                "pos": np.zeros(6),
                "vel": np.zeros(6),
                "torque": np.zeros(6),
            }
        state = self._loop.get_joint_state()
        return {
            "pos": np.array(state.pos),
            "vel": np.array(state.vel),
            "torque": np.array(state.torque),
        }

    def get_gripper_state(self) -> dict[str, float]:
        """Return normalized gripper position and torque."""
        if self._loop is None:
            return {"pos": 0.0, "torque": 0.0}
        state = self._loop.get_gripper_state()
        return {"pos": state.pos, "torque": state.torque}

    def get_perf(self):
        """Return performance snapshot from the RT loop."""
        if self._loop is None:
            return None
        return self._loop.get_perf()

    def read_cycle_times(self, n: int = 10000) -> np.ndarray:
        """Return array of recent cycle times in microseconds."""
        if self._loop is None:
            return np.array([], dtype=np.float32)
        return self._loop.read_cycle_times(n)
