#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT="$SCRIPT_DIR"

MODE="${1:-final}"

case "$MODE" in
    test50)
        BASE="${PROJECT}/mcmc_watchdog/test_50G"
        ;;
    final)
        BASE="${PROJECT}/mcmc_watchdog/production"
        ;;
    *)
        echo "Usage:"
        echo "  $0 test50"
        echo "  $0 final"
        exit 1
        ;;
esac

PIDFILE="$BASE/watchdog.pid"

if [ ! -s "$PIDFILE" ]; then
    echo "No PID file found."
    exit 0
fi

PID=$(cat "$PIDFILE")

if kill -0 "$PID" 2>/dev/null; then

    echo "Stopping watchdog PID=$PID"
    kill -TERM "$PID"

    for i in {1..10}; do
        kill -0 "$PID" 2>/dev/null || break
        sleep 1
    done

    if kill -0 "$PID" 2>/dev/null; then
        echo "Watchdog did not stop; sending KILL."
        kill -KILL "$PID"
    fi

else
    echo "Watchdog PID $PID is no longer running."
fi

rm -f "$PIDFILE"
