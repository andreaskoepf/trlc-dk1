"""Minimal V4L2 control access via raw ioctls.

We deliberately avoid pulling in `v4l-utils` (not installed on the Jetson) or
extra Python packages. The Video4Linux2 control ABI is stable and small enough
to handle directly: four ioctls cover everything we need.

The control names returned here match what `uvcdynctrl` / Guvcview show, so
profile JSON keys are human-readable.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
from dataclasses import dataclass, field
from typing import Optional


# --- ioctl encoding (matches asm-generic/ioctl.h) ----------------------------

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


def _IOC(direction: int, type_: int, nr: int, size: int) -> int:
    return (
        (direction << (_IOC_NRBITS + _IOC_TYPEBITS + _IOC_SIZEBITS))
        | (size << (_IOC_NRBITS + _IOC_TYPEBITS))
        | (type_ << _IOC_NRBITS)
        | nr
    )


def _IOR(type_: int, nr: int, size: int) -> int:
    return _IOC(_IOC_READ, type_, nr, size)


def _IOWR(type_: int, nr: int, size: int) -> int:
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, size)


# --- V4L2 structs (linux/videodev2.h) ----------------------------------------

_V4L2_IOC_TYPE = ord("V")


class v4l2_queryctrl(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("name", ctypes.c_char * 32),
        ("minimum", ctypes.c_int32),
        ("maximum", ctypes.c_int32),
        ("step", ctypes.c_int32),
        ("default_value", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class v4l2_control(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("value", ctypes.c_int32),
    ]


class _v4l2_querymenu_union(ctypes.Union):
    _fields_ = [
        ("name", ctypes.c_char * 32),
        ("value", ctypes.c_int64),
    ]


class v4l2_querymenu(ctypes.Structure):
    _pack_ = 1
    _anonymous_ = ("_u",)
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("index", ctypes.c_uint32),
        ("_u", _v4l2_querymenu_union),
        ("reserved", ctypes.c_uint32),
    ]


VIDIOC_QUERYCTRL = _IOWR(_V4L2_IOC_TYPE, 36, ctypes.sizeof(v4l2_queryctrl))
VIDIOC_G_CTRL = _IOWR(_V4L2_IOC_TYPE, 27, ctypes.sizeof(v4l2_control))
VIDIOC_S_CTRL = _IOWR(_V4L2_IOC_TYPE, 28, ctypes.sizeof(v4l2_control))
VIDIOC_QUERYMENU = _IOWR(_V4L2_IOC_TYPE, 37, ctypes.sizeof(v4l2_querymenu))

# Flags / types we care about.
V4L2_CTRL_FLAG_NEXT_CTRL = 0x80000000
V4L2_CTRL_FLAG_DISABLED = 0x0001
V4L2_CTRL_FLAG_READ_ONLY = 0x0004
V4L2_CTRL_FLAG_INACTIVE = 0x0010
V4L2_CTRL_FLAG_WRITE_ONLY = 0x0040

V4L2_CTRL_TYPE_INTEGER = 1
V4L2_CTRL_TYPE_BOOLEAN = 2
V4L2_CTRL_TYPE_MENU = 3
V4L2_CTRL_TYPE_BUTTON = 4
V4L2_CTRL_TYPE_INTEGER64 = 5
V4L2_CTRL_TYPE_CTRL_CLASS = 6
V4L2_CTRL_TYPE_STRING = 7
V4L2_CTRL_TYPE_BITMASK = 8
V4L2_CTRL_TYPE_INTEGER_MENU = 9

_TYPE_NAMES = {
    V4L2_CTRL_TYPE_INTEGER: "int",
    V4L2_CTRL_TYPE_BOOLEAN: "bool",
    V4L2_CTRL_TYPE_MENU: "menu",
    V4L2_CTRL_TYPE_BUTTON: "button",
    V4L2_CTRL_TYPE_INTEGER64: "int64",
    V4L2_CTRL_TYPE_CTRL_CLASS: "class",
    V4L2_CTRL_TYPE_STRING: "string",
    V4L2_CTRL_TYPE_BITMASK: "bitmask",
    V4L2_CTRL_TYPE_INTEGER_MENU: "int_menu",
}


# --- High-level dataclasses --------------------------------------------------


@dataclass
class ControlInfo:
    id: int
    name: str
    type: str  # one of _TYPE_NAMES values
    minimum: int
    maximum: int
    step: int
    default: int
    flags: int
    menu: Optional[list[tuple[int, str]]] = field(default=None)

    @property
    def is_writable(self) -> bool:
        if self.flags & V4L2_CTRL_FLAG_DISABLED:
            return False
        if self.flags & V4L2_CTRL_FLAG_READ_ONLY:
            return False
        if self.type in ("class", "button"):
            return False
        return True


# --- ioctl helpers -----------------------------------------------------------


def _ioctl(fd: int, request: int, arg) -> None:
    # ctypes ioctl: pass the struct directly; fcntl mutates it in-place.
    fcntl.ioctl(fd, request, arg)


def query_menu(fd: int, ctrl_id: int, lo: int, hi: int) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    for idx in range(lo, hi + 1):
        q = v4l2_querymenu()
        q.id = ctrl_id
        q.index = idx
        try:
            _ioctl(fd, VIDIOC_QUERYMENU, q)
        except OSError as e:
            if e.errno in (errno.EINVAL, errno.ENODATA):
                continue
            raise
        try:
            name = q.name.decode("utf-8", errors="replace").rstrip("\x00")
        except Exception:
            name = ""
        items.append((idx, name))
    return items


def enumerate_controls(device_path: str) -> list[ControlInfo]:
    """List all V4L2 controls exposed by `device_path`, in driver order."""
    fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
    try:
        out: list[ControlInfo] = []
        ctrl_id = V4L2_CTRL_FLAG_NEXT_CTRL  # iterate via NEXT_CTRL
        seen: set[int] = set()
        while True:
            q = v4l2_queryctrl()
            q.id = ctrl_id
            try:
                _ioctl(fd, VIDIOC_QUERYCTRL, q)
            except OSError as e:
                # EINVAL is the spec'd terminator. The Innomaker UVC driver
                # returns EIO once enumeration is exhausted instead; treat that
                # as end-of-list too once we've seen at least one control.
                if e.errno in (errno.EINVAL, errno.ENODATA):
                    break
                if e.errno == errno.EIO and out:
                    break
                raise
            if q.id in seen:
                break
            seen.add(q.id)
            type_name = _TYPE_NAMES.get(q.type, f"unknown_{q.type}")
            try:
                name = q.name.decode("utf-8", errors="replace").rstrip("\x00")
            except Exception:
                name = f"ctrl_0x{q.id:08x}"
            info = ControlInfo(
                id=q.id,
                name=name,
                type=type_name,
                minimum=q.minimum,
                maximum=q.maximum,
                step=q.step,
                default=q.default_value,
                flags=q.flags,
            )
            if type_name in ("menu", "int_menu") and not (q.flags & V4L2_CTRL_FLAG_DISABLED):
                try:
                    info.menu = query_menu(fd, q.id, q.minimum, q.maximum)
                except OSError:
                    info.menu = None
            out.append(info)
            ctrl_id = q.id | V4L2_CTRL_FLAG_NEXT_CTRL
        return out
    finally:
        os.close(fd)


def get_control(fd_or_path, ctrl_id: int) -> int:
    own = isinstance(fd_or_path, str)
    fd = os.open(fd_or_path, os.O_RDWR | os.O_NONBLOCK) if own else fd_or_path
    try:
        c = v4l2_control()
        c.id = ctrl_id
        _ioctl(fd, VIDIOC_G_CTRL, c)
        return c.value
    finally:
        if own:
            os.close(fd)


def set_control(fd_or_path, ctrl_id: int, value: int) -> None:
    own = isinstance(fd_or_path, str)
    fd = os.open(fd_or_path, os.O_RDWR | os.O_NONBLOCK) if own else fd_or_path
    try:
        c = v4l2_control()
        c.id = ctrl_id
        c.value = int(value)
        _ioctl(fd, VIDIOC_S_CTRL, c)
    finally:
        if own:
            os.close(fd)


def snapshot_values(device_path: str) -> dict[str, int]:
    """Return {control_name: current_value} for all writable controls."""
    fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
    try:
        out: dict[str, int] = {}
        for info in enumerate_controls(device_path):
            if not info.is_writable:
                continue
            try:
                out[info.name] = get_control(fd, info.id)
            except OSError:
                continue
        return out
    finally:
        os.close(fd)


def control_index(device_path: str) -> dict[str, ControlInfo]:
    """{name: ControlInfo} for quick lookup by JSON-profile key."""
    return {c.name: c for c in enumerate_controls(device_path)}
