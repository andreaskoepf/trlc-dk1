from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class GravityCompensator:
    """
    Computes gravity compensation torques using MuJoCo inverse dynamics.

    Only the first `num_dofs` joints of the model are used (arm joints).
    The gripper is excluded — its gravity contribution is negligible and
    it operates in force-position (EMIT) mode.

    The URDF must contain ``<mujoco><compiler strippath="false"/></mujoco>``
    so that MuJoCo can resolve relative mesh paths.

    Args:
        model_path: Path to a MuJoCo XML (.xml) or URDF (.urdf) file.
        num_dofs:   Number of arm joints to compute torques for (default 6).
    """

    def __init__(self, model_path: str, num_dofs: int = 6) -> None:
        try:
            import mujoco
        except ImportError as e:
            raise ImportError(
                "mujoco is required for gravity compensation. "
                "Install it with: pip install mujoco"
            ) from e

        self._mujoco = mujoco
        self.num_dofs = num_dofs

        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        if self.mj_model.nq < num_dofs:
            raise ValueError(
                f"MuJoCo model has {self.mj_model.nq} DoFs but num_dofs={num_dofs}"
            )

        logger.info(
            "GravityCompensator loaded: %s (%d DoF model, using first %d)",
            model_path,
            self.mj_model.nq,
            num_dofs,
        )

    def compute(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation torques for the arm joints.

        Args:
            q: Joint positions (radians), shape (num_dofs,) or larger.

        Returns:
            tau_grav: Gravity torques, shape (num_dofs,).
        """
        mujoco = self._mujoco
        self.mj_data.qpos[: self.num_dofs] = q[: self.num_dofs]
        self.mj_data.qvel[:] = 0.0
        self.mj_data.qacc[:] = 0.0
        # Use mj_forward + qfrc_bias instead of mj_inverse + qfrc_inverse.
        # mj_inverse includes constraint forces from joint limits which can
        # produce wildly incorrect torques near limit boundaries.
        # qfrc_bias with qvel=0 gives pure gravity torques.
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.mj_data.qfrc_bias[: self.num_dofs].copy()


class NoGravityComp:
    """Drop-in replacement when gravity compensation is disabled."""

    num_dofs: int = 6

    def compute(self, q: np.ndarray) -> np.ndarray:
        return np.zeros(self.num_dofs)
