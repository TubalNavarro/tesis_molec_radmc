#!/usr/bin/env bash

###############################################################################
# MCMC MEMORY WATCHDOG
#
# Detecta automáticamente:
#   MCMC_master.py
#   multiprocessing workers
#   radmc3d y demás descendientes
#
# Acciones:
#   1) proceso individual supera límite -> mata solamente ese MCMC
#   2) todos los MCMC superan límite total -> mata todos los MCMC
#   3) MemAvailable del nodo baja del límite -> mata todos los MCMC
#
# Antes de matar guarda información forense.
###############################################################################

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT="$SCRIPT_DIR"

USER_NAME=$(id -un)

INTERVAL="${INTERVAL:-1}"

PROCESS_LIMIT_GIB="${PROCESS_LIMIT_GIB:-10}"
TOTAL_LIMIT_GIB="${TOTAL_LIMIT_GIB:-150}"
NODE_MIN_AVAILABLE_GIB="${NODE_MIN_AVAILABLE_GIB:-150}"

TOTAL_CONSECUTIVE_REQUIRED="${TOTAL_CONSECUTIVE_REQUIRED:-2}"
NODE_CONSECUTIVE_REQUIRED="${NODE_CONSECUTIVE_REQUIRED:-2}"

DRY_RUN="${DRY_RUN:-0}"

BASE_DIR="${MCMC_WATCHDOG_DIR:-${PROJECT}/mcmc_watchdog/production}"

mkdir -p "$BASE_DIR" || {
    echo "ERROR: cannot create $BASE_DIR" >&2
    exit 1
}

LOG="${BASE_DIR}/watchdog.log"
PIDFILE="${BASE_DIR}/watchdog.pid"
LOCKFILE="${BASE_DIR}/watchdog.lock"

touch "$LOG" || exit 1

###############################################################################
# Prevent duplicate watchdogs in the same mode/directory
###############################################################################

command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required." >&2
    exit 1
}

exec 9>"$LOCKFILE"

if ! flock -n 9; then
    echo "Another watchdog is already using $BASE_DIR" >&2
    exit 1
fi

echo $$ > "$PIDFILE"

###############################################################################
# Convert limits to KiB
###############################################################################

PROCESS_LIMIT_KB=$(( PROCESS_LIMIT_GIB * 1024 * 1024 ))
TOTAL_LIMIT_KB=$(( TOTAL_LIMIT_GIB * 1024 * 1024 ))
NODE_MIN_AVAILABLE_KB=$(( NODE_MIN_AVAILABLE_GIB * 1024 * 1024 ))

###############################################################################

log()
{
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"
}

cleanup()
{
    rm -f "$PIDFILE"
    log "Watchdog stopped PID=$$"
}

trap 'exit 0' TERM INT
trap cleanup EXIT

###############################################################################
# Find MCMC masters
#
# Workers inherit the MCMC_master.py command line, so pgrep alone cannot
# distinguish master from worker.
#
# We call "master" a MCMC_master.py process whose parent is NOT another
# MCMC_master.py process.
###############################################################################

get_masters()
{
    local -a candidates
    local -A is_candidate
    local p parent

    mapfile -t candidates < <(
        pgrep -u "$USER_NAME" -f 'MCMC_master\.py' 2>/dev/null
    )

    [ "${#candidates[@]}" -gt 0 ] || return

    for p in "${candidates[@]}"; do
        is_candidate["$p"]=1
    done

    for p in "${candidates[@]}"; do

        [ -r "/proc/$p/status" ] || continue

        parent=$(awk '/^PPid:/ {print $2}' "/proc/$p/status" 2>/dev/null)

        if [ -z "${is_candidate[$parent]+x}" ]; then
            echo "$p"
        fi
    done
}

###############################################################################
# Recursively obtain process tree
###############################################################################

collect_tree()
{
    local root="$1"

    local -a queue
    local p child

    queue=("$root")

    while [ "${#queue[@]}" -gt 0 ]; do

        p="${queue[0]}"
        queue=("${queue[@]:1}")

        [ -d "/proc/$p" ] || continue

        echo "$p"

        if [ -r "/proc/$p/task/$p/children" ]; then
            for child in $(cat "/proc/$p/task/$p/children" 2>/dev/null); do
                queue+=("$child")
            done
        fi
    done
}

###############################################################################
# Forensic snapshot
###############################################################################

capture_incident()
{
    local reason="$1"
    local suspect_pid="$2"

    local stamp dir p

    stamp=$(date '+%Y%m%d_%H%M%S')
    dir="${BASE_DIR}/incident_${stamp}"

    mkdir -p "$dir"

    {
        echo "TIME=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "HOST=$(hostname)"
        echo "USER=$USER_NAME"
        echo "REASON=$reason"
        echo "SUSPECT_PID=$suspect_pid"
        echo
        echo "PROCESS_LIMIT_GIB=$PROCESS_LIMIT_GIB"
        echo "TOTAL_LIMIT_GIB=$TOTAL_LIMIT_GIB"
        echo "NODE_MIN_AVAILABLE_GIB=$NODE_MIN_AVAILABLE_GIB"
    } > "$dir/trigger.txt"

    free -h > "$dir/free.txt" 2>&1

    cat /proc/meminfo > "$dir/meminfo.txt" 2>&1
    cat /proc/vmstat  > "$dir/vmstat.txt" 2>&1

    {
        echo "===== CPU PRESSURE ====="
        cat /proc/pressure/cpu 2>/dev/null

        echo
        echo "===== MEMORY PRESSURE ====="
        cat /proc/pressure/memory 2>/dev/null

        echo
        echo "===== IO PRESSURE ====="
        cat /proc/pressure/io 2>/dev/null
    } > "$dir/pressure.txt"

    ps -eo \
        user,pid,ppid,pgid,sid,stat,rss,vsz,%cpu,%mem,etime,cmd \
        --sort=-rss \
        > "$dir/all_processes.txt" 2>&1

    mapfile -t masters < <(get_masters)

    {
        echo "MCMC masters: ${masters[*]}"
        echo

        printf "%-10s %-10s %-10s %-14s %s\n" \
            PID PPID PGID RSS_KiB CMD

        declare -A seen

        for m in "${masters[@]}"; do

            while read -r p; do

                [ -n "$p" ] || continue
                [ -z "${seen[$p]+x}" ] || continue

                seen["$p"]=1

                [ -r "/proc/$p/status" ] || continue

                parent=$(awk '/^PPid:/ {print $2}' "/proc/$p/status")
                rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$p/status")
                pgid=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
                cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)

                printf "%-10s %-10s %-10s %-14s %s\n" \
                    "$p" "$parent" "$pgid" "${rss:-0}" "$cmd"

            done < <(collect_tree "$m")
        done

    } > "$dir/mcmc_processes.txt"

    ###########################################################################
    # Suspect process
    ###########################################################################

    if [ "$suspect_pid" -gt 0 ] 2>/dev/null &&
       [ -d "/proc/$suspect_pid" ]; then

        mkdir -p "$dir/suspect_${suspect_pid}"

        cat "/proc/$suspect_pid/status" \
            > "$dir/suspect_${suspect_pid}/status.txt" \
            2>/dev/null || true

        cat "/proc/$suspect_pid/smaps_rollup" \
            > "$dir/suspect_${suspect_pid}/smaps_rollup.txt" \
            2>/dev/null || true

        cat "/proc/$suspect_pid/maps" \
            > "$dir/suspect_${suspect_pid}/maps.txt" \
            2>/dev/null || true

        tr '\0' ' ' < "/proc/$suspect_pid/cmdline" \
            > "$dir/suspect_${suspect_pid}/cmdline.txt" \
            2>/dev/null || true

        ls -l "/proc/$suspect_pid/fd" \
            > "$dir/suspect_${suspect_pid}/fds.txt" \
            2>/dev/null || true
    fi

    ###########################################################################
    # Also capture the largest user processes
    ###########################################################################

    mapfile -t top_pids < <(
        ps -u "$USER_NAME" -o pid=,rss= --sort=-rss |
        head -n 8 |
        awk '{print $1}'
    )

    for p in "${top_pids[@]}"; do

        [ -d "/proc/$p" ] || continue

        mkdir -p "$dir/pid_$p"

        cat "/proc/$p/status" \
            > "$dir/pid_$p/status.txt" \
            2>/dev/null || true

        cat "/proc/$p/smaps_rollup" \
            > "$dir/pid_$p/smaps_rollup.txt" \
            2>/dev/null || true

        tr '\0' ' ' < "/proc/$p/cmdline" \
            > "$dir/pid_$p/cmdline.txt" \
            2>/dev/null || true

    done

    log "FORENSIC SNAPSHOT saved=$dir"
}

###############################################################################
# Kill one MCMC tree
###############################################################################

kill_mcmc_tree()
{
    local master="$1"

    [ -d "/proc/$master" ] || return

    mapfile -t pids < <(collect_tree "$master")

    log "TERM master=$master tree_size=${#pids[@]}"

    # Children first, master last
    for ((i=${#pids[@]}-1; i>=0; i--)); do
        kill -TERM "${pids[$i]}" 2>/dev/null || true
    done

    sleep 5

    if [ -d "/proc/$master" ]; then

        mapfile -t pids < <(collect_tree "$master")

        log "KILL master=$master survivors=${#pids[@]}"

        for ((i=${#pids[@]}-1; i>=0; i--)); do
            kill -KILL "${pids[$i]}" 2>/dev/null || true
        done
    fi
}

###############################################################################
# Kill all MCMC
###############################################################################

kill_all_mcmc()
{
    mapfile -t masters < <(get_masters)

    if [ "${#masters[@]}" -eq 0 ]; then
        return
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        log "DRY_RUN would kill masters=${masters[*]}"
        return
    fi

    declare -A all_pids

    ###########################################################################
    # First collect every process before killing anything
    ###########################################################################

    for m in "${masters[@]}"; do

        while read -r p; do
            [ -n "$p" ] && all_pids["$p"]=1
        done < <(collect_tree "$m")

    done

    log "TERM ALL_MCMC masters=${masters[*]} processes=${#all_pids[@]}"

    ###########################################################################
    # TERM every process immediately
    ###########################################################################

    for p in "${!all_pids[@]}"; do
        kill -TERM "$p" 2>/dev/null || true
    done

    ###########################################################################
    # Single grace period
    ###########################################################################

    sleep 5

    ###########################################################################
    # KILL survivors
    ###########################################################################

    survivors=0

    for p in "${!all_pids[@]}"; do

        if kill -0 "$p" 2>/dev/null; then
            survivors=$(( survivors + 1 ))
            kill -KILL "$p" 2>/dev/null || true
        fi

    done

    if [ "$survivors" -gt 0 ]; then
        log "KILL ALL_MCMC survivors=$survivors"
    else
        log "All MCMC processes terminated after TERM"
    fi
}

###############################################################################
# Kill the MCMC containing a suspect PID
###############################################################################

kill_owner_mcmc()
{
    local suspect="$1"

    mapfile -t masters < <(get_masters)

    for m in "${masters[@]}"; do

        while read -r p; do

            if [ "$p" = "$suspect" ]; then

                if [ "$DRY_RUN" -eq 1 ]; then
                    log "DRY_RUN would kill master=$m because PID=$suspect exceeded limit"
                else
                    kill_mcmc_tree "$m"
                fi

                return
            fi

        done < <(collect_tree "$m")
    done

    log "Could not resolve owner of PID=$suspect; killing all MCMC"
    kill_all_mcmc
}

###############################################################################
# MAIN LOOP
###############################################################################

log "============================================================"
log "Watchdog started PID=$$"
log "process_limit=${PROCESS_LIMIT_GIB}GiB total_limit=${TOTAL_LIMIT_GIB}GiB node_min=${NODE_MIN_AVAILABLE_GIB}GiB"
log "interval=${INTERVAL}s dry_run=${DRY_RUN}"

total_bad_count=0
node_bad_count=0
heartbeat_count=0

while true; do

    mapfile -t masters < <(get_masters)

    if [ "${#masters[@]}" -eq 0 ]; then

        total_bad_count=0
        node_bad_count=0

        sleep "$INTERVAL"
        continue
    fi

    declare -A all_pids

    for m in "${masters[@]}"; do

        while read -r p; do
            [ -n "$p" ] && all_pids["$p"]=1
        done < <(collect_tree "$m")

    done

    total_rss_kb=0
    largest_rss_kb=0
    largest_pid=0
    suspect_pid=0

    for p in "${!all_pids[@]}"; do

        [ -r "/proc/$p/status" ] || continue

        rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$p/status" 2>/dev/null)
        rss=${rss:-0}

        total_rss_kb=$(( total_rss_kb + rss ))

        if [ "$rss" -gt "$largest_rss_kb" ]; then
            largest_rss_kb="$rss"
            largest_pid="$p"
        fi

        if [ "$rss" -ge "$PROCESS_LIMIT_KB" ]; then
            suspect_pid="$p"
            break
        fi
    done

    ###########################################################################
    # Individual-process emergency
    ###########################################################################

    if [ "$suspect_pid" -ne 0 ]; then

        suspect_rss=$(awk -v x="$largest_rss_kb" \
            'BEGIN {printf "%.2f",x/1048576}')

        reason="INDIVIDUAL_PROCESS_LIMIT PID=${suspect_pid} RSS=${suspect_rss}GiB"

        log "TRIGGER $reason"

        capture_incident "$reason" "$suspect_pid"

        kill_owner_mcmc "$suspect_pid"

        total_bad_count=0
        node_bad_count=0

        sleep "$INTERVAL"
        continue
    fi

    ###########################################################################
    # Node memory
    ###########################################################################

    mem_available_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    mem_available_kb=${mem_available_kb:-0}

    ###########################################################################
    # Consecutive total-memory trigger
    ###########################################################################

    if [ "$total_rss_kb" -ge "$TOTAL_LIMIT_KB" ]; then
        total_bad_count=$(( total_bad_count + 1 ))
    else
        total_bad_count=0
    fi

    ###########################################################################
    # Consecutive node-memory trigger
    ###########################################################################

    if [ "$mem_available_kb" -le "$NODE_MIN_AVAILABLE_KB" ]; then
        node_bad_count=$(( node_bad_count + 1 ))
    else
        node_bad_count=0
    fi

    ###########################################################################
    # Total MCMC emergency
    ###########################################################################

    if [ "$total_bad_count" -ge "$TOTAL_CONSECUTIVE_REQUIRED" ]; then

        total_gib=$(awk -v x="$total_rss_kb" \
            'BEGIN {printf "%.2f",x/1048576}')

        reason="TOTAL_MCMC_LIMIT RSS=${total_gib}GiB"

        log "TRIGGER $reason"

        capture_incident "$reason" "$largest_pid"

        kill_all_mcmc

        total_bad_count=0
        node_bad_count=0

        sleep "$INTERVAL"
        continue
    fi

    ###########################################################################
    # Node-memory emergency
    ###########################################################################

    if [ "$node_bad_count" -ge "$NODE_CONSECUTIVE_REQUIRED" ]; then

        available_gib=$(awk -v x="$mem_available_kb" \
            'BEGIN {printf "%.2f",x/1048576}')

        reason="LOW_NODE_MEMORY MemAvailable=${available_gib}GiB"

        log "TRIGGER $reason"

        capture_incident "$reason" "$largest_pid"

        kill_all_mcmc

        total_bad_count=0
        node_bad_count=0

        sleep "$INTERVAL"
        continue
    fi

    ###########################################################################
    # Heartbeat approximately once per minute while MCMC is active
    ###########################################################################

    heartbeat_count=$(( heartbeat_count + 1 ))

    if [ "$heartbeat_count" -ge 60 ]; then

        total_gib=$(awk -v x="$total_rss_kb" \
            'BEGIN {printf "%.2f",x/1048576}')

        available_gib=$(awk -v x="$mem_available_kb" \
            'BEGIN {printf "%.2f",x/1048576}')

        largest_gib=$(awk -v x="$largest_rss_kb" \
            'BEGIN {printf "%.2f",x/1048576}')

        log "ACTIVE masters=${#masters[@]} total=${total_gib}GiB largest_pid=${largest_pid} largest=${largest_gib}GiB MemAvailable=${available_gib}GiB"

        heartbeat_count=0
    fi

    sleep "$INTERVAL"

done
