"""Camera profile JSON: load, save, apply.

Profile shape (one file covers all cameras in a recording session):

    {
      "name": "indoor_default",
      "description": "...",
      "cameras": {
        "WRIST_LEFT":  { "Brightness": 0, "Contrast": 52, ... },
        "WRIST_RIGHT": { "Brightness": 0, ... },
        "CONTEXT_CAM": { ... }
      }
    }

Camera keys are the same env-var names used in `port_config.env`, so the
profile is portable between machines as long as port_config is regenerated.

Control keys are the human-readable V4L2 names returned by `enumerate_controls`
(e.g. "Brightness", "White Balance Temperature"). This is stable per UVC
device class — all Innomaker U30CAM-4K-S1 units expose the same set.

`apply_profile` is idempotent: setting a control to a value it already has is
a no-op, and unknown keys are logged + ignored (so adding new fields in the
future doesn't break older recordings).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .v4l2_controls import (
    ControlInfo,
    control_index,
    get_control,
    set_control,
    snapshot_values,
)

log = logging.getLogger(__name__)


# Order in which we set controls. White-balance / auto-exposure flags must be
# applied BEFORE the related manual values, because turning auto-WB on
# clobbers the manual White Balance Temperature, and Aperture-Priority mode
# overrides the Exposure Time value.
_APPLY_ORDER = (
    "White Balance, Automatic",
    "Auto Exposure",
    "Power Line Frequency",
    "Backlight Compensation",
    "Brightness",
    "Contrast",
    "Saturation",
    "Hue",
    "Gamma",
    "Sharpness",
    "Gain",
    "White Balance Temperature",
    "Exposure Time, Absolute",
    "Exposure, Dynamic Framerate",
)


@dataclass
class CameraProfile:
    name: str = "untitled"
    description: str = ""
    cameras: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "cameras": self.cameras,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraProfile":
        if "cameras" not in d:
            raise ValueError("profile missing 'cameras' key")
        return cls(
            name=d.get("name", "untitled"),
            description=d.get("description", ""),
            cameras={k: dict(v) for k, v in d["cameras"].items()},
        )


def load_profile(path: os.PathLike | str) -> CameraProfile:
    with open(path) as f:
        return CameraProfile.from_dict(json.load(f))


def save_profile(profile: CameraProfile, path: os.PathLike | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)
        f.write("\n")
    os.replace(tmp, p)


def snapshot_to_profile(
    name: str,
    description: str,
    device_paths: dict[str, str],
) -> CameraProfile:
    """Read current values from each device and pack into a profile."""
    return CameraProfile(
        name=name,
        description=description,
        cameras={
            cam_key: snapshot_values(path)
            for cam_key, path in device_paths.items()
        },
    )


def _sort_apply_order(names: list[str]) -> list[str]:
    pri = {n: i for i, n in enumerate(_APPLY_ORDER)}
    return sorted(names, key=lambda n: pri.get(n, 10_000))


def apply_to_device(
    device_path: str,
    settings: dict[str, int],
    *,
    skip_unchanged: bool = True,
) -> dict[str, tuple[int, int]]:
    """Apply `settings` to one device. Returns {name: (old, new)} for changes
    that were actually written. Unknown keys are skipped with a warning.
    """
    idx = control_index(device_path)
    changes: dict[str, tuple[int, int]] = {}
    fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
    try:
        for name in _sort_apply_order(list(settings.keys())):
            info: Optional[ControlInfo] = idx.get(name)
            if info is None:
                log.warning("Unknown control %r on %s — skipping", name, device_path)
                continue
            if not info.is_writable:
                log.warning("Control %r is not writable on %s — skipping", name, device_path)
                continue
            target = int(settings[name])
            target = max(info.minimum, min(info.maximum, target))
            try:
                current = get_control(fd, info.id)
            except OSError as e:
                log.warning("Could not read %r on %s: %s", name, device_path, e)
                current = None
            if skip_unchanged and current == target:
                continue
            try:
                set_control(fd, info.id, target)
            except OSError as e:
                # Setting a manual exposure when Auto Exposure is set to
                # Aperture Priority returns EBUSY — that's expected; the user
                # should set Auto Exposure first. Log and continue.
                log.warning("Failed to set %r=%d on %s: %s", name, target, device_path, e)
                continue
            changes[name] = (current if current is not None else -1, target)
    finally:
        os.close(fd)
    return changes


def apply_profile(
    profile: CameraProfile,
    device_paths: dict[str, str],
    *,
    skip_unchanged: bool = True,
) -> dict[str, dict[str, tuple[int, int]]]:
    """Apply `profile` across the given {cam_key: device_path} mapping.

    Returns nested {cam_key: {control_name: (old, new)}} of actual changes.
    Cameras present in `device_paths` but missing from the profile are left
    untouched; cameras in the profile but missing from `device_paths` are
    logged + skipped.
    """
    results: dict[str, dict[str, tuple[int, int]]] = {}
    for cam_key, settings in profile.cameras.items():
        path = device_paths.get(cam_key)
        if not path:
            log.warning("Profile camera %r has no matching device path — skipping", cam_key)
            continue
        if not os.path.exists(path):
            log.warning("Device path %s missing for %r — skipping", path, cam_key)
            continue
        results[cam_key] = apply_to_device(path, settings, skip_unchanged=skip_unchanged)
    return results


def reset_to_defaults(device_path: str) -> dict[str, tuple[int, int]]:
    """Set every writable control on the device back to its driver default."""
    idx = control_index(device_path)
    defaults = {
        info.name: info.default
        for info in idx.values()
        if info.is_writable
    }
    return apply_to_device(device_path, defaults, skip_unchanged=True)


def load_device_paths_from_env(env_path: os.PathLike | str) -> dict[str, str]:
    """Parse port_config.env and return {WRIST_LEFT/WRIST_RIGHT/CONTEXT_CAM: /dev/...}."""
    keys = ("WRIST_LEFT", "WRIST_RIGHT", "CONTEXT_CAM")
    out: dict[str, str] = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k in keys:
                out[k] = v
    return out
