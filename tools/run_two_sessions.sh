#!/bin/bash
# Run restart_sg_nav.sh twice, each for a maximum of 5 hours.
# The entire orchestrator runs detached via nohup so it survives terminal close.
#
# Usage:
#   ./tools/run_two_sessions.sh [restart_sg_nav.sh flags...]
#   Example: ./tools/run_two_sessions.sh --inject --variant 0
#
# This script immediately backgrounds itself. Progress is logged to:
#   tools/two_sessions_<timestamp>.log

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${SCRIPT_DIR}/two_sessions_${TIMESTAMP}.log"

# If not already running under nohup, re-exec detached
if [ -z "$_TWO_SESSIONS_DETACHED" ]; then
    export _TWO_SESSIONS_DETACHED=1
    nohup bash "$0" "$@" > "$LOGFILE" 2>&1 &
    ORCHESTRATOR_PID=$!
    echo "Orchestrator launched in background (PID $ORCHESTRATOR_PID)"
    echo "Log: $LOGFILE"
    echo "Monitor: tail -f $LOGFILE"
    exit 0
fi

# ---- Detached orchestrator starts here ----

RESTART_SCRIPT="${SCRIPT_DIR}/restart_sg_nav.sh"
DURATION=27000  # 5 hours in seconds
NUM_RUNS=1

echo "$(date): Starting $NUM_RUNS sessions, $((DURATION/3600))h each"
echo "$(date): Flags passed to restart_sg_nav.sh: $*"

for RUN in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "========================================"
    echo "$(date): SESSION $RUN/$NUM_RUNS — starting"
    echo "========================================"

    # Launch the restart script (which starts SG_Nav.py via nohup)
    bash "$RESTART_SCRIPT" "$@"
    sleep 5  # give it time to start

    # Find the SG_Nav.py PID
    PID=$(pgrep -f "python SG_Nav.py" | head -1)
    if [ -z "$PID" ]; then
        echo "$(date): WARNING — SG_Nav.py did not start. Skipping session $RUN."
        continue
    fi
    echo "$(date): SG_Nav.py running with PID $PID"

    # Wait up to DURATION seconds, checking every 60s if process is still alive
    ELAPSED=0
    INTERVAL=60
    while [ $ELAPSED -lt $DURATION ]; do
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "$(date): SG_Nav.py (PID $PID) finished on its own after ~${ELAPSED}s"
            break
        fi
    done

    # If still running after the timeout, kill it
    if kill -0 "$PID" 2>/dev/null; then
        echo "$(date): Time limit reached ($((DURATION/3600))h). Killing PID $PID..."
        kill "$PID" 2>/dev/null
        sleep 5
        # Force kill if still alive
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null
            echo "$(date): Force-killed PID $PID"
        fi
        echo "$(date): SESSION $RUN complete (timed out)"
    else
        echo "$(date): SESSION $RUN complete (natural exit)"
    fi

    # Brief pause between sessions
    if [ "$RUN" -lt "$NUM_RUNS" ]; then
        echo "$(date): Pausing 10s before next session..."
        sleep 10
    fi
done

echo ""
echo "$(date): All $NUM_RUNS sessions done."
