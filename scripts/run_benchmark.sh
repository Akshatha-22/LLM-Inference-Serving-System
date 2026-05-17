#!/bin/bash
# scripts/run_benchmark.sh
# Day 6-7: Run load test and save results

echo "=========================================="
echo "LLM Inference Server - Load Test"
echo "=========================================="

# Check if server is running
echo "Checking if server is running on http://localhost:8000..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q 200
if [ $? -ne 0 ]; then
    echo "❌ Server is not running!"
    echo "   Please start the server first: python run_server.py"
    exit 1
fi
echo "✅ Server is running"

# Parameters
USERS=${1:-20}
SPAWN_RATE=${2:-2}
RUN_TIME=${3:-60s}

echo ""
echo "Running load test with:"
echo "  Users: $USERS"
echo "  Spawn rate: $SPAWN_RATE users/second"
echo "  Duration: $RUN_TIME"
echo ""

# Run locust in headless mode
.venv\Scripts\locust -f benchmarks/locustfile.py \
    --host=http://localhost:8000 \
    --users $USERS \
    --spawn-rate $SPAWN_RATE \
    --run-time $RUN_TIME \
    --headless \
    --only-summary \
    --html benchmarks/results/report.html

echo ""
echo "=========================================="
echo "✅ Load test complete!"
echo "   Results saved to: benchmarks/results/baseline_results.json"
echo "   HTML report: benchmarks/results/report.html"
echo "=========================================="