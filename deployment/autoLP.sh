#!/bin/bash

# USB mode is selected in src/main.py. The proven launcher behavior is kept
# unchanged so the detection pipeline, dashboard, and systemd lifecycle remain
# identical to the LAN-camera and video modes.

set -u

PROJECT_ROOT="/home/pi/NanoPi-R6S-ALPCMR"
DASHBOARD_ROOT="/home/pi/car-detector"
PYTHON_BIN="/home/pi/lp/bin/python3.10"


find_node_bin() {
    local candidate

    if command -v node >/dev/null 2>&1; then
        command -v node
        return 0
    fi

    # systemd does not load the interactive shell setup used by NVM/FNM/ASDF.
    # Check their common per-user install locations explicitly.
    for candidate in \
        /usr/local/bin/node \
        /usr/bin/node \
        /home/pi/.nvm/versions/node/*/bin/node \
        /home/pi/.local/share/fnm/node-versions/*/installation/bin/node \
        /home/pi/.asdf/shims/node
    do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}


start_headless() {
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "[autoLP] Python not found or not executable: $PYTHON_BIN" >&2
        return 1
    fi
    if [ ! -f "$PROJECT_ROOT/src/main.py" ]; then
        echo "[autoLP] Pipeline entry point not found" >&2
        return 1
    fi
    if [ ! -f "$DASHBOARD_ROOT/app.js" ]; then
        echo "[autoLP] Dashboard entry point not found" >&2
        return 1
    fi
    node_bin="$(find_node_bin)" || node_bin=""
    if [ -z "$node_bin" ]; then
        echo "[autoLP] Node.js was not found in PATH or common user install locations" >&2
        return 1
    fi

    echo "[autoLP] Node.js: $node_bin ($("$node_bin" --version))"

    pipeline_pid=""
    dashboard_pid=""

    cleanup() {
        trap - EXIT INT TERM
        if [ -n "$pipeline_pid" ] && kill -0 "$pipeline_pid" 2>/dev/null; then
            kill -TERM "$pipeline_pid" 2>/dev/null || true
        fi
        if [ -n "$dashboard_pid" ] && kill -0 "$dashboard_pid" 2>/dev/null; then
            kill -TERM "$dashboard_pid" 2>/dev/null || true
        fi
        wait 2>/dev/null || true
    }

    trap cleanup EXIT
    trap 'exit 0' INT TERM

    echo "[autoLP] Starting detection pipeline..."
    (
        cd "$PROJECT_ROOT" || exit 1
        exec "$PYTHON_BIN" src/main.py --headless
    ) &
    pipeline_pid=$!

    echo "[autoLP] Starting dashboard..."
    (
        cd "$DASHBOARD_ROOT" || exit 1
        exec "$node_bin" app.js
    ) &
    dashboard_pid=$!

    echo "[autoLP] Pipeline PID: $pipeline_pid"
    echo "[autoLP] Dashboard PID: $dashboard_pid"

    # If either required process exits, end the wrapper. systemd will clean up
    # the remaining process and restart the complete pair.
    wait -n "$pipeline_pid" "$dashboard_pid"
}


start_desktop() {
    # Interactive/manual mode: preserve the existing terminal-based workflow.
    gnome-terminal -- bash -c "cd '$PROJECT_ROOT' && '$PYTHON_BIN' src/main.py; exec bash"
    gnome-terminal -- bash -c "cd '$DASHBOARD_ROOT' && bash run.sh; exec bash"
}


case "${1:-}" in
    --headless)
        start_headless
        ;;
    "")
        start_desktop
        ;;
    *)
        echo "Usage: $0 [--headless]" >&2
        exit 2
        ;;
esac
