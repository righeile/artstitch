#!/bin/bash
# Double-click this file in Finder to start ArtStitch.
# It opens in your browser; close this Terminal window to stop the app.

cd "$(dirname "$0")" || exit 1

hold() {
    echo
    echo "Press Return to close this window."
    read -r _
    exit 1
}

# Prefer a project virtualenv if one exists, otherwise whatever python3 is on PATH.
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    echo "Could not find python3."
    echo "Install Python 3, or create a virtualenv here with:"
    echo "    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    hold
fi

echo "ArtStitch"
echo "Using: $("$PY" -c 'import sys; print(sys.executable)') ($("$PY" -V 2>&1))"
echo

missing=$("$PY" - <<'PYEOF'
import importlib.util
need = {"numpy": "numpy", "scipy": "scipy", "cv2": "opencv-python", "tifffile": "tifffile",
        "imagecodecs": "imagecodecs", "PIL": "pillow", "flask": "flask"}
print(" ".join(pkg for mod, pkg in need.items()
               if not importlib.util.find_spec(mod)))
PYEOF
)
if [ -n "$missing" ]; then
    echo "Missing package(s): $missing"
    printf "Install them now? [y/N] "
    read -r reply
    case "$reply" in
        [yY]*) "$PY" -m pip install -r requirements.txt || hold ;;
        *) echo "Install them with:  $PY -m pip install -r requirements.txt"; hold ;;
    esac
fi

SERVER=app.py
PORT=${PORT:-8800}

# A copy still running on the port is stopped first, so the page never talks to
# an older server behind the current files. A process is ours if it is a Python
# whose command line names this folder's server, however it was started: from
# here, from the parent, by full path.
ours() {
    local target cwd
    target="$(pwd -P)/$SERVER"
    case "$(basename "$(ps -o comm= -p "$1")")" in [Pp]ython*) ;; *) return 1 ;; esac
    cwd=$(lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')
    case " $(ps -o command= -p "$1") " in *" ${target#"$cwd"/} "*) ;; *) return 1 ;; esac
}

# Stop our old copy; step past a port some other program holds.
for _ in $(seq 20); do
    pid=$(lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1)
    [ -z "$pid" ] && break
    if ! ours "$pid"; then PORT=$((PORT + 1)); continue; fi
    echo "Stopping the copy already running on port $PORT (pid $pid)."
    pkill -P "$pid"; kill "$pid"
    for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    kill -9 "$pid" 2>/dev/null
done

URL="http://127.0.0.1:$PORT"
echo "Starting the app on $URL"
echo "Leave this window open while you work; close it to stop the app."
echo

"$PY" "$SERVER" --port "$PORT" &
APP_PID=$!
trap 'kill "$APP_PID" 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 60); do
    if curl -sf "$URL/api/health" >/dev/null 2>&1; then
        open "$URL"
        break
    fi
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        echo "The app stopped while starting up. See the messages above."
        hold
    fi
    sleep 1
done

wait "$APP_PID"
