#!/bin/bash
# Shut down the GLM-5.1 vLLM server (and optionally Ray).
#
#   bash stop_glm.sh          # stop ONLY the vLLM API server; leave Ray up
#                             # (so serve_glm.sh can relaunch without a Ray
#                             #  restart — serve_glm.sh does NOT start Ray).
#   bash stop_glm.sh --ray    # also `ray stop` on BOTH nodes
#
# WHY the narrow kill pattern: a broad `pkill -f python` / `-f vllm` also
# matches raylet and its bash wrapper and would drop the worker node's Ray
# worker (see docs/GOTCHAS.md). We match the exact driver command line instead.
set -uo pipefail

WORKER="${WORKER_HOST:?ssh-reachable worker node hostname}"
RAY_BIN="${RAY_BIN:?path to the venv ray binary (same path on both nodes)}"
# Exact api_server command (started by serve_glm.sh via this interpreter).
SERVER_PAT='python -m vllm[.]entrypoints[.]openai[.]api_server'

stop_ray=0
[[ "${1:-}" == "--ray" ]] && stop_ray=1

echo "[stop] terminating vLLM API server (SIGTERM)..."
pkill -f "$SERVER_PAT" 2>/dev/null || true

# Wait up to ~40s for a clean exit, then SIGKILL any stragglers.
for i in $(seq 1 20); do
    n=$(pgrep -f "$SERVER_PAT" | wc -l)
    [[ "$n" -eq 0 ]] && break
    sleep 2
done
if pgrep -f "$SERVER_PAT" > /dev/null; then
    echo "[stop] still alive after 40s; SIGKILL..."
    pkill -9 -f "$SERVER_PAT" 2>/dev/null || true
    sleep 3
fi
n=$(pgrep -f "$SERVER_PAT" | wc -l)
echo "[stop] api_server processes remaining: $n"

# Also reap any orphaned EngineCore/RayWorkerProc children the driver spawned
# on THIS node (they exit with the driver, but be defensive). Match only the
# vLLM engine core module, never raylet.
pkill -f 'python -m vllm[.]v1[.]engine[.]core' 2>/dev/null || true

if [[ "$stop_ray" -eq 1 ]]; then
    echo "[stop] stopping Ray on the worker ($WORKER)..."
    ssh "$WORKER" "$RAY_BIN stop" 2>&1 | sed 's/^/  [worker] /' || true
    echo "[stop] stopping Ray on the head..."
    "$RAY_BIN" stop 2>&1 | sed 's/^/  [head] /' || true
else
    echo "[stop] Ray left running. GPU reservation should now read 0.0/16.0:"
    "$RAY_BIN" status 2>&1 | grep -E "GPU" | head -1 | sed 's/^/  /' || true
    echo "[stop] (pass --ray to also stop Ray on both nodes)"
fi

echo "[stop] GPU memory now:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1 | sed 's/^/  head gpu0: /'
echo "[stop] done."
