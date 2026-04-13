#!/bin/bash
# Auto-restart script for SG_Nav.py
# Checks if SG_Nav.py is running; if not, restarts it with a timestamped log file.
#
# Usage:
#   ./restart_sg_nav.sh                                    # baseline with frontier_teleport
#   ./restart_sg_nav.sh --inject                           # inject from default manifest
#   ./restart_sg_nav.sh --inject --manifest path/to.jsonl  # inject from custom manifest
#   ./restart_sg_nav.sh --inject --variant 1               # use variant B (default: 0)
#   ./restart_sg_nav.sh --inject --record 0                # use specific manifest record
#   ./restart_sg_nav.sh --no-teleport                      # disable frontier_teleport

PROJECT_DIR="/home/kalyan/20-Credits/SG-Nav"
CONDA_ENV="SG_Nav"
PYTHON="/home/kalyan/.conda/envs/SG_Nav/bin/python"
STATE_FILE="${PROJECT_DIR}/tools/.llm_escape_state"

# Defaults
USE_INJECT="off"
MANIFEST_FILE="data/dynamicqa/manifest.jsonl"
INJECT_VARIANT=0
INJECT_RECORD=""
USE_FRONTIER_TELEPORT="on"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --inject)
            USE_INJECT="on"; shift ;;
        --manifest)
            MANIFEST_FILE="$2"; shift 2 ;;
        --variant)
            INJECT_VARIANT="$2"; shift 2 ;;
        --record)
            INJECT_RECORD="$2"; shift 2 ;;
        --no-teleport)
            USE_FRONTIER_TELEPORT="off"; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--inject] [--manifest FILE] [--variant N] [--record N] [--no-teleport]"
            exit 1 ;;
    esac
done

cd "$PROJECT_DIR" || exit 1

# Check if SG_Nav.py is already running
if pgrep -f "python SG_Nav.py" > /dev/null 2>&1; then
    echo "$(date): SG_Nav.py is still running, skipping restart."
else
    echo "$USE_FRONTIER_TELEPORT" > "$STATE_FILE"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    FLAGS="--visualize"
    LOG_SUFFIX="baseline"

    # Frontier teleport
    if [ "$USE_FRONTIER_TELEPORT" = "on" ]; then
        FLAGS="$FLAGS --frontier_teleport"
        LOG_SUFFIX="random_frontier_escape"
    fi

    # Object injection
    if [ "$USE_INJECT" = "on" ]; then
        FLAGS="$FLAGS --inject_manifest ${MANIFEST_FILE} --inject_variant ${INJECT_VARIANT}"
        if [ -n "$INJECT_RECORD" ]; then
            FLAGS="$FLAGS --inject_record_idx ${INJECT_RECORD}"
        fi
        LOG_SUFFIX="${LOG_SUFFIX}_inject"
    fi

    LOGFILE="${PROJECT_DIR}/output_ext_${TIMESTAMP}_${LOG_SUFFIX}.log"

    echo "$(date): SG_Nav.py not running, restarting with flags: ${FLAGS}"
    echo "$(date): Log file: ${LOGFILE}"

    # Activate conda environment variables (LD_LIBRARY_PATH etc.)
    export PATH="/home/kalyan/.conda/envs/SG_Nav/bin:/home/kalyan/cuda-11.8/bin:$PATH"
    export LD_LIBRARY_PATH="/home/kalyan/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"
    export CHALLENGE_CONFIG_FILE="configs/challenge_objectnav2021.local.rgbd.yaml"

    nohup "$PYTHON" SG_Nav.py $FLAGS > "$LOGFILE" 2>&1 &
    echo "$(date): Started SG_Nav.py with PID $! (teleport=$USE_FRONTIER_TELEPORT, inject=$USE_INJECT)"
fi
