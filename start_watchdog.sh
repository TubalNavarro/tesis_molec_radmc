#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT="$SCRIPT_DIR"

MODE="${1:-final}"

case "$MODE" in

    test50)
        BASE="${PROJECT}/mcmc_watchdog/test_50G"

        PROCESS_LIMIT_GIB=100
        TOTAL_LIMIT_GIB=50
        NODE_MIN_AVAILABLE_GIB=1
        ;;

    final)
        BASE="${PROJECT}/mcmc_watchdog/production"

        PROCESS_LIMIT_GIB=10
        TOTAL_LIMIT_GIB=180
        NODE_MIN_AVAILABLE_GIB=150
        ;;

    *)
        echo "Usage:"
        echo "  $0 test50"
        echo "  $0 final"
        exit 1
        ;;
esac

mkdir -p "$BASE"

###############################################################################
# Detect existing watchdog for this mode
###############################################################################

if [ -s "$BASE/watchdog.pid" ]; then

    OLD_PID=$(cat "$BASE/watchdog.pid")

    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Watchdog already running."
        echo "PID: $OLD_PID"
        echo "Directory: $BASE"
        exit 0
    else
        echo "Removing stale PID file."
        rm -f "$BASE/watchdog.pid"
    fi
fi

###############################################################################
# Start detached
###############################################################################

nohup setsid env \
    MCMC_WATCHDOG_DIR="$BASE" \
    PROCESS_LIMIT_GIB="$PROCESS_LIMIT_GIB" \
    TOTAL_LIMIT_GIB="$TOTAL_LIMIT_GIB" \
    NODE_MIN_AVAILABLE_GIB="$NODE_MIN_AVAILABLE_GIB" \
    INTERVAL=1 \
    TOTAL_CONSECUTIVE_REQUIRED=2 \
    NODE_CONSECUTIVE_REQUIRED=2 \
    DRY_RUN=0 \
    "${PROJECT}/mcmc_memory_watchdog.sh" \
    > "${BASE}/nohup_watchdog.out" \
    2>&1 < /dev/null &

PID=$!

sleep 2

if kill -0 "$PID" 2>/dev/null; then

    echo "Watchdog started successfully."
    echo "Mode: $MODE"
    echo "PID: $PID"
    echo "Directory: $BASE"

else

    echo "ERROR: watchdog did not remain running."
    echo
    cat "${BASE}/nohup_watchdog.out"
    exit 1

fi
