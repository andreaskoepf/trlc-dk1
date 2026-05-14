"""Camera profile tool: query/apply UVC camera controls + a small web GUI.

Submodules:
    v4l2_controls — raw V4L2 ioctl wrappers (QUERYCTRL/G_CTRL/S_CTRL/QUERYMENU).
    profile        — JSON profile load/save/apply.
    app            — local HTTP server + MJPEG preview + REST UI.
"""
