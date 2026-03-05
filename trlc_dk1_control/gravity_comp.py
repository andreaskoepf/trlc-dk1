from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import numpy as np

logger = logging.getLogger(__name__)


def _urdf_strip_meshes(urdf_path: str) -> str:
    """
    Return a URDF XML string with all mesh geometry replaced by a tiny sphere.

    MuJoCo's URDF parser strips the directory prefix from mesh filenames, which
    breaks loading when meshes live in subdirectories.  For inverse dynamics we
    only need mass/inertia/joint data — geometry is irrelevant — so we swap every
    <mesh> element for a <sphere radius="0.001"/> to satisfy MuJoCo's parser.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for geometry in root.iter("geometry"):
        mesh = geometry.find("mesh")
        if mesh is not None:
            geometry.remove(mesh)
            ET.SubElement(geometry, "sphere", {"radius": "0.001"})

    return ET.tostring(root, encoding="unicode")


class GravityCompensator:
    """
    Computes gravity compensation torques using MuJoCo inverse dynamics.

    Only the first `num_dofs` joints of the model are used (arm joints).
    The gripper is excluded — its gravity contribution is negligible and
    it operates in force-position (EMIT) mode.

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

        if model_path.endswith(".urdf"):
            xml_str = _urdf_strip_meshes(model_path)
            self.mj_model = mujoco.MjModel.from_xml_string(xml_str)
        else:
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
        mujoco.mj_inverse(self.mj_model, self.mj_data)
        return self.mj_data.qfrc_inverse[: self.num_dofs].copy()


class NoGravityComp:
    """Drop-in replacement when gravity compensation is disabled."""

    num_dofs: int = 6

    def compute(self, q: np.ndarray) -> np.ndarray:
        return np.zeros(self.num_dofs)
