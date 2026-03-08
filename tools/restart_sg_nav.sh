#!/bin/bash
# Auto-restart script for SG_Nav.py
# Checks if SG_Nav.py is running; if not, restarts it with a timestamped log file.
# Alternates between --llm_escape enabled and disabled using a state file.

PROJECT_DIR="/home/kalyan/20-Credits/SG-Nav"
CONDA_ENV="SG_Nav"
PYTHON="/home/kalyan/.conda/envs/SG_Nav/bin/python"
STATE_FILE="${PROJECT_DIR}/tools/.llm_escape_state"

cd "$PROJECT_DIR" || exit 1

# Check if SG_Nav.py is already running
if pgrep -f "python SG_Nav.py" > /dev/null 2>&1; then
    echo "$(date): SG_Nav.py is still running, skipping restart."
else
    # Determine whether to use --llm_escape this run
    # State file contains "on" or "off"; starts with "off" if missing
    # if [ -f "$STATE_FILE" ] && [ "$(cat "$STATE_FILE")" = "off" ]; then
    #     USE_RANDOM_FRONTIER_ESCAPE="on"
    # else
    #     USE_RANDOM_FRONTIER_ESCAPE="off"
    # fi

    USE_RANDOM_FRONTIER_ESCAPE="on"
    echo "$USE_RANDOM_FRONTIER_ESCAPE" > "$STATE_FILE"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    if [ "$USE_RANDOM_FRONTIER_ESCAPE" = "on" ]; then
        FLAGS="--visualize --frontier_teleport"
        LOGFILE="${PROJECT_DIR}/output_ext_${TIMESTAMP}_random_frontier_escape.log"
    else
        FLAGS="--visualize"
        LOGFILE="${PROJECT_DIR}/output_ext_${TIMESTAMP}_baseline.log"
    fi

    echo "$(date): SG_Nav.py not running, restarting with flags: ${FLAGS}"
    echo "$(date): Log file: ${LOGFILE}"

    # Activate conda environment variables (LD_LIBRARY_PATH etc.)
    export PATH="/home/kalyan/.conda/envs/SG_Nav/bin:/home/kalyan/cuda-11.8/bin:$PATH"
    export LD_LIBRARY_PATH="/home/kalyan/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"
    export CHALLENGE_CONFIG_FILE="configs/challenge_objectnav2021.local.rgbd.yaml"

    nohup "$PYTHON" SG_Nav.py $FLAGS > "$LOGFILE" 2>&1 &
    echo "$(date): Started SG_Nav.py with PID $! (random_frontier_escape=$USE_RANDOM_FRONTIER_ESCAPE)"
fi
