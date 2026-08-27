#!/usr/bin/env bash
set -u

if [ -z "${BASH_VERSINFO+x}" ]; then
    echo "emel-model-parity-inventory-watchdog: Bash is required" >&2
    exit 125
fi

usage() {
    echo "usage: run-with-deadline.sh --seconds N --grace-seconds N -- COMMAND [ARG ...]" >&2
    exit 2
}

is_positive_integer() {
    case "$1" in
        ''|*[!0-9]*|0) return 1 ;;
        *) return 0 ;;
    esac
}

[ "$#" -ge 6 ] || usage
[ "$1" = "--seconds" ] || usage
seconds="$2"
shift 2
[ "$1" = "--grace-seconds" ] || usage
grace_seconds="$2"
shift 2
[ "$1" = "--" ] || usage
shift
[ "$#" -gt 0 ] || usage
is_positive_integer "$seconds" || usage
is_positive_integer "$grace_seconds" || usage

capture_dir="$(mktemp -d "${TMPDIR:-/tmp}/emel-model-watchdog.XXXXXX")" || {
    echo "emel-model-parity-inventory-watchdog: cannot create capture directory" >&2
    exit 125
}
stdout_file="$capture_dir/stdout"
stderr_file="$capture_dir/stderr"
timeout_marker="$capture_dir/timed-out"
start_gate="$capture_dir/start-target"
child_pid=''
group_pid=''
timer_pid=''

group_exists() {
    [ -n "$group_pid" ] && kill -0 -- "-$group_pid" 2>/dev/null
}

verify_group_gone() {
    local attempt
    for ((attempt = 0; attempt < 100; attempt += 1)); do
        if ! group_exists; then
            group_pid=''
            return 0
        fi
        sleep 0.01
    done
    return 1
}

terminate_group() {
    kill -TERM -- "-$group_pid" 2>/dev/null || true
    sleep "$grace_seconds"
    kill -KILL -- "-$group_pid" 2>/dev/null || true
    verify_group_gone
}

cleanup() {
    if [ -n "$timer_pid" ]; then
        kill -KILL "$timer_pid" 2>/dev/null || true
        wait "$timer_pid" 2>/dev/null || true
    fi
    if [ -n "$group_pid" ]; then
        kill -KILL -- "-$group_pid" 2>/dev/null || true
    fi
    if [ -n "$child_pid" ]; then
        wait "$child_pid" 2>/dev/null || true
    fi
    rm -rf "$capture_dir"
}
trap cleanup 0
trap 'exit 125' HUP INT TERM

# Bash monitor mode makes the parent shell create a process group for each asynchronous job before
# that job can execute. A shell-owned start gate keeps the target from entering user code until the
# parent has verified that the exact child-PID process group exists.
if ! set -m; then
    echo "emel-model-parity-inventory-watchdog: Bash monitor mode is unavailable" >&2
    exit 125
fi
(
    while [[ ! -e "$start_gate" ]]; do
        :
    done
    exec "$@"
) >"$stdout_file" 2>"$stderr_file" &
child_pid=$!
group_pid=$child_pid

if ! kill -0 -- "-$child_pid" 2>/dev/null; then
    echo "emel-model-parity-inventory-watchdog: target process group isolation failed (pid=$child_pid)" >&2
    exit 125
fi

(
    sleep "$seconds"
    : >"$timeout_marker"
    kill -TERM -- "-$group_pid" 2>/dev/null || true
    sleep "$grace_seconds"
    kill -KILL -- "-$group_pid" 2>/dev/null || true
) &
timer_pid=$!
: >"$start_gate"

wait "$child_pid" 2>/dev/null
child_status=$?
child_pid=''

if [ -f "$timeout_marker" ]; then
    # Let the timer complete the promised grace period and SIGKILL attempt before classifying the
    # result. This also reaps the timer rather than leaving deferred watchdog work behind.
    wait "$timer_pid" 2>/dev/null || true
    timer_pid=''
    if ! verify_group_gone; then
        cat "$stdout_file"
        cat "$stderr_file" >&2
        echo "emel-model-parity-inventory-watchdog: timeout cleanup failed; target process group remains (pid=$group_pid)" >&2
        exit 125
    fi
    cat "$stdout_file"
    cat "$stderr_file" >&2
    echo "emel-model-parity-inventory-watchdog: timeout after $seconds second(s); sent SIGTERM then SIGKILL to the process group after a $grace_seconds second grace" >&2
    exit 124
fi

kill -KILL "$timer_pid" 2>/dev/null || true
wait "$timer_pid" 2>/dev/null || true
timer_pid=''
if group_exists; then
    if ! terminate_group; then
        cat "$stdout_file"
        cat "$stderr_file" >&2
        echo "emel-model-parity-inventory-watchdog: descendant cleanup failed after target exit; process group remains (pid=$group_pid)" >&2
        exit 125
    fi
    cat "$stdout_file"
    cat "$stderr_file" >&2
    echo "emel-model-parity-inventory-watchdog: target exited while descendants remained; sent SIGTERM then SIGKILL to the process group after a $grace_seconds second grace" >&2
    exit 125
fi
group_pid=''
cat "$stdout_file"
cat "$stderr_file" >&2
exit "$child_status"
