#!/bin/bash

# Bimanual teleoperation with two DK-1 arms using RT impedance control.
# Sources port_config.env for stable device paths.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../port_config.env"

uv run python examples/bi_teleop.py \
    --left-leader "$LEFT_LEADER" \
    --right-leader "$RIGHT_LEADER" \
    --left-follower "$LEFT_FOLLOWER" \
    --right-follower "$RIGHT_FOLLOWER" \
    --mode rt_impedance
