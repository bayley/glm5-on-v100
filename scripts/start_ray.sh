#!/bin/bash
# Start the 2-node Ray cluster for GLM-5.1 (head + worker, 8 GPUs each).
# serve_glm.sh does NOT start Ray; run this first (once per boot / after a crash).
#
#   bash start_ray.sh          # (re)start Ray on both nodes, verify 0.0/16.0 GPU
#   bash start_ray.sh --keep   # skip if a healthy 16-GPU cluster is already up
#
# Required environment (no sane defaults exist — they are cluster-specific):
#   RAY_BIN        path to the ray binary INSIDE the venv that has this repo +
#                  the compiled SM70 extensions + the cuBLAS 12.9 fix
#   WORKER_HOST    ssh-reachable name of the worker node
#   HEAD_IP        this (head) node's IP as seen by the worker
#   HEAD_IFNAME    head node's ethernet ifname (GLOO/NCCL sockets)
#   HEAD_IB_HCA    head node's IB HCA (e.g. mlx5_1)
#   WORKER_IP / WORKER_IFNAME / WORKER_IB_HCA   same, for the worker node
#
# CRITICAL (see docs/GOTCHAS.md):
#  - raylet hardcodes its python-worker command to whatever `ray` binary started
#    it, so RAY_BIN must be the venv ray on BOTH nodes.
#  - NCCL/GLOO interface names DIFFER per node and must be exported BEFORE
#    `ray start` so they propagate into the Ray actors.
#  - A crashed run can leave a stale placement group reserving all 16 GPUs
#    ("0.0 used of 16.0 reserved"); a clean stop+start clears it.
set -uo pipefail

RAY_BIN="${RAY_BIN:?path to the venv ray binary (same path on both nodes)}"
WORKER="${WORKER_HOST:?ssh-reachable worker node hostname}"

HEAD_IP="${HEAD_IP:?head node IP}"
HEAD_PORT="${HEAD_PORT:-6379}"
HEAD_IFNAME="${HEAD_IFNAME:?head node ethernet ifname}"
HEAD_IB_HCA="${HEAD_IB_HCA:?head node IB HCA}"
WORKER_IP="${WORKER_IP:?worker node IP}"
WORKER_IFNAME="${WORKER_IFNAME:?worker node ethernet ifname}"
WORKER_IB_HCA="${WORKER_IB_HCA:?worker node IB HCA}"

keep=0
[[ "${1:-}" == "--keep" ]] && keep=1

healthy() {
    # 0 == a 16-GPU cluster is up with nothing reserved.
    "$RAY_BIN" status 2>/dev/null | grep -qE '0\.0/16\.0 GPU'
}

if [[ "$keep" -eq 1 ]] && healthy; then
    echo "[ray] healthy 16-GPU cluster already up (--keep); nothing to do."
    "$RAY_BIN" status 2>&1 | grep -E "GPU" | head -1 | sed 's/^/  /'
    exit 0
fi

echo "[ray] stopping any existing Ray on both nodes (clears stale placement groups)..."
ssh "$WORKER" "$RAY_BIN stop" 2>&1 | sed 's/^/  [worker] /' || true
"$RAY_BIN" stop 2>&1 | sed 's/^/  [head] /' || true
sleep 3

echo "[ray] starting HEAD ($HEAD_IP:$HEAD_PORT, ifname=$HEAD_IFNAME)..."
VLLM_HOST_IP="$HEAD_IP" \
GLOO_SOCKET_IFNAME="$HEAD_IFNAME" NCCL_SOCKET_IFNAME="$HEAD_IFNAME" \
NCCL_IB_HCA="$HEAD_IB_HCA" NCCL_IB_GID_INDEX=0 \
    "$RAY_BIN" start --head --node-ip-address="$HEAD_IP" \
        --port="$HEAD_PORT" --num-gpus=8 2>&1 | sed 's/^/  [head] /'
sleep 3

echo "[ray] starting WORKER on $WORKER ($WORKER_IP, ifname=$WORKER_IFNAME)..."
ssh "$WORKER" "VLLM_HOST_IP=$WORKER_IP \
GLOO_SOCKET_IFNAME=$WORKER_IFNAME NCCL_SOCKET_IFNAME=$WORKER_IFNAME \
NCCL_IB_HCA=$WORKER_IB_HCA NCCL_IB_GID_INDEX=0 \
$RAY_BIN start --address=$HEAD_IP:$HEAD_PORT \
--node-ip-address=$WORKER_IP --num-gpus=8" 2>&1 | sed 's/^/  [worker] /'

echo "[ray] waiting for both nodes to register 16 GPUs..."
ok=0
for i in $(seq 1 20); do
    if "$RAY_BIN" status 2>/dev/null | grep -qE '/16\.0 GPU'; then ok=1; break; fi
    sleep 2
done

echo "[ray] final status:"
"$RAY_BIN" status 2>&1 | grep -E "GPU|node_" | sed 's/^/  /'
if [[ "$ok" -eq 1 ]] && healthy; then
    echo "[ray] OK: 16 GPUs available (0.0/16.0). Now run: bash serve_glm.sh"
else
    echo "[ray] WARNING: cluster not cleanly at 0.0/16.0 GPU. Check both nodes" >&2
    echo "[ray]          ('$RAY_BIN status'); a stale reservation may remain." >&2
    exit 1
fi
