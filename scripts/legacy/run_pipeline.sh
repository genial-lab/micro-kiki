#!/bin/bash
# Full pipeline: distill python -> stop teacher -> train stacks -> restart teacher -> router -> E2E
# Run on kxkm-ai after reasoning distill is done.
set -e

VENV=/home/kxkm/KIKI-models-tuning/.venv/bin/python
cd /home/kxkm/micro-kiki
export UNSLOTH_COMPILE_DISABLE=1
export PYTHONUNBUFFERED=1

TEACHER_PID=$(pgrep -f "llama-server.*8000" | head -1)
TEACHER_CMD=$(ps -p $TEACHER_PID -o args= 2>/dev/null)
echo "Teacher PID: $TEACHER_PID"
echo "Teacher CMD: $TEACHER_CMD"

echo "=== STEP 1: Distill Python ==="
$VENV scripts/distill_fast.py --domain python --max-examples 800 --max-tokens 1024 2>&1 | tee /tmp/distill-python.log
echo "=== STEP 1 DONE ==="

echo "=== Stopping teacher for GPU training ==="
kill $TEACHER_PID 2>/dev/null || true
sleep 5
echo "Teacher stopped. GPU free."

echo "=== STEP 2: Train stack-02 reasoning ==="
$VENV scripts/train_stack_kxkm.py --domain reasoning --stack-id 02 2>&1 | tee /tmp/train-stack02.log
echo "=== STEP 2 DONE ==="

echo "=== STEP 3: Train stack-03 python ==="
$VENV scripts/train_stack_kxkm.py --domain python --stack-id 03 2>&1 | tee /tmp/train-stack03.log
echo "=== STEP 3 DONE ==="

echo "=== STEP 4: Train Router v0 ==="
$VENV scripts/train_router_kxkm.py 2>&1 | tee /tmp/train-router.log
echo "=== STEP 4 DONE ==="

echo "=== Restarting teacher ==="
nohup $TEACHER_CMD > /tmp/teacher-restart.log 2>&1 &
sleep 10
echo "Teacher restarted. Waiting for it to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "Teacher is ready!"
        break
    fi
    sleep 5
done

echo "=== STEP 5: E2E Smoke Test ==="
$VENV scripts/smoke_e2e.py 2>&1 | tee /tmp/smoke-e2e.log
echo "=== STEP 5 DONE ==="

echo "=== ALL STEPS COMPLETE ==="
