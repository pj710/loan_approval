"""
Phase 10: Performance Benchmarking Script

Measures API latency, throughput, and model inference times.
"""
import requests
import time
import statistics
import json
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import sys


# Configuration
API_BASE_URL = "http://localhost:8000"
LATENCY_THRESHOLD_MS = 500
PROJECT_ROOT = Path(__file__).parent.parent


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def measure_latency(endpoint: str, payload: Dict[Any, Any], n_requests: int = 50) -> Dict[str, float]:
    """
    Measure latency statistics for an endpoint.
    
    Returns dict with mean, median, p95, p99, min, max latencies in ms.
    """
    latencies = []
    errors = 0
    
    for i in range(n_requests):
        start = time.perf_counter()
        try:
            if endpoint.startswith("/health"):
                response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
            else:
                response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=30)
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            if response.status_code == 200:
                latencies.append(elapsed_ms)
            else:
                errors += 1
        except requests.exceptions.RequestException:
            errors += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{n_requests}", end="\r")
    
    print()  # New line after progress
    
    if not latencies:
        return {"error": "All requests failed"}
    
    return {
        "count": len(latencies),
        "errors": errors,
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        "p99": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
        "min": min(latencies),
        "max": max(latencies)
    }


def measure_throughput(endpoint: str, payload: Dict[Any, Any], duration_seconds: int = 10) -> Dict[str, float]:
    """
    Measure throughput (requests per second) for an endpoint.
    """
    completed = 0
    errors = 0
    start_time = time.perf_counter()
    
    while (time.perf_counter() - start_time) < duration_seconds:
        try:
            response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=30)
            if response.status_code == 200:
                completed += 1
            else:
                errors += 1
        except requests.exceptions.RequestException:
            errors += 1
    
    elapsed = time.perf_counter() - start_time
    
    return {
        "completed": completed,
        "errors": errors,
        "duration_seconds": elapsed,
        "requests_per_second": completed / elapsed if elapsed > 0 else 0
    }


def measure_concurrent_load(endpoint: str, payload: Dict[Any, Any], n_workers: int = 5, n_requests_per_worker: int = 20) -> Dict[str, Any]:
    """
    Measure performance under concurrent load.
    """
    all_latencies = []
    errors = 0
    
    def worker_task():
        latencies = []
        worker_errors = 0
        for _ in range(n_requests_per_worker):
            start = time.perf_counter()
            try:
                response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=30)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if response.status_code == 200:
                    latencies.append(elapsed_ms)
                else:
                    worker_errors += 1
            except:
                worker_errors += 1
        return latencies, worker_errors
    
    start_time = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(worker_task) for _ in range(n_workers)]
        for future in concurrent.futures.as_completed(futures):
            latencies, worker_errors = future.result()
            all_latencies.extend(latencies)
            errors += worker_errors
    
    elapsed = time.perf_counter() - start_time
    
    if not all_latencies:
        return {"error": "All requests failed"}
    
    return {
        "workers": n_workers,
        "total_requests": n_workers * n_requests_per_worker,
        "completed": len(all_latencies),
        "errors": errors,
        "total_duration_seconds": elapsed,
        "mean_latency_ms": statistics.mean(all_latencies),
        "p95_latency_ms": statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 20 else max(all_latencies),
        "requests_per_second": len(all_latencies) / elapsed if elapsed > 0 else 0
    }


def run_benchmarks():
    """Run all performance benchmarks."""
    print("=" * 70)
    print("PHASE 10: PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_BASE_URL}")
    print(f"Latency Threshold: {LATENCY_THRESHOLD_MS}ms")
    print()
    
    # Check API availability
    print("Checking API availability...")
    if not check_api_health():
        print("❌ API is not available. Start with: uvicorn src.api.main:app --port 8000")
        sys.exit(1)
    print("✅ API is healthy\n")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "api_url": API_BASE_URL,
        "threshold_ms": LATENCY_THRESHOLD_MS,
        "benchmarks": {}
    }
    
    # Standard payload
    standard_payload = {
        'loan_amount': 250000,
        'property_value': 300000,
        'income': 80000,
        'interest_rate': 6.5,
        'loan_term': 360
    }
    
    # 1. Health Endpoint Latency
    print("=" * 50)
    print("1. Health Endpoint Latency")
    print("=" * 50)
    health_results = measure_latency("/health", {}, n_requests=30)
    results["benchmarks"]["health_endpoint"] = health_results
    
    if "error" not in health_results:
        print(f"  Mean:   {health_results['mean']:.2f} ms")
        print(f"  Median: {health_results['median']:.2f} ms")
        print(f"  P95:    {health_results['p95']:.2f} ms")
        print(f"  Max:    {health_results['max']:.2f} ms")
        status = "✅ PASS" if health_results['p95'] < 100 else "❌ FAIL"
        print(f"  Result: {status} (target: <100ms)")
    print()
    
    # 2. Predict Endpoint Latency
    print("=" * 50)
    print("2. Predict Endpoint Latency")
    print("=" * 50)
    predict_results = measure_latency("/predict", standard_payload, n_requests=50)
    results["benchmarks"]["predict_endpoint"] = predict_results
    
    if "error" not in predict_results:
        print(f"  Mean:   {predict_results['mean']:.2f} ms")
        print(f"  Median: {predict_results['median']:.2f} ms")
        print(f"  P95:    {predict_results['p95']:.2f} ms")
        print(f"  P99:    {predict_results['p99']:.2f} ms")
        print(f"  Max:    {predict_results['max']:.2f} ms")
        status = "✅ PASS" if predict_results['p95'] < LATENCY_THRESHOLD_MS else "❌ FAIL"
        print(f"  Result: {status} (target: <{LATENCY_THRESHOLD_MS}ms)")
    print()
    
    # 3. Explain Endpoint Latency
    print("=" * 50)
    print("3. Explain Endpoint Latency")
    print("=" * 50)
    explain_results = measure_latency("/explain", standard_payload, n_requests=20)
    results["benchmarks"]["explain_endpoint"] = explain_results
    
    if "error" not in explain_results:
        print(f"  Mean:   {explain_results['mean']:.2f} ms")
        print(f"  Median: {explain_results['median']:.2f} ms")
        print(f"  P95:    {explain_results['p95']:.2f} ms")
        print(f"  Max:    {explain_results['max']:.2f} ms")
        # Explain can be slower due to SHAP
        status = "✅ PASS" if explain_results['p95'] < 2000 else "⚠️ SLOW"
        print(f"  Result: {status} (target: <2000ms)")
    print()
    
    # 4. Batch Predict Latency (10 applications)
    print("=" * 50)
    print("4. Batch Predict Latency (10 applications)")
    print("=" * 50)
    batch_payload = {"applications": [standard_payload] * 10}
    batch_results = measure_latency("/batch/predict", batch_payload, n_requests=20)
    results["benchmarks"]["batch_predict_10"] = batch_results
    
    if "error" not in batch_results:
        print(f"  Mean:   {batch_results['mean']:.2f} ms")
        print(f"  Median: {batch_results['median']:.2f} ms")
        print(f"  P95:    {batch_results['p95']:.2f} ms")
        print(f"  Max:    {batch_results['max']:.2f} ms")
        avg_per_app = batch_results['mean'] / 10
        print(f"  Avg per application: {avg_per_app:.2f} ms")
    print()
    
    # 5. Throughput Test
    print("=" * 50)
    print("5. Throughput Test (10 seconds)")
    print("=" * 50)
    throughput_results = measure_throughput("/predict", standard_payload, duration_seconds=10)
    results["benchmarks"]["throughput"] = throughput_results
    
    print(f"  Completed: {throughput_results['completed']} requests")
    print(f"  Errors:    {throughput_results['errors']}")
    print(f"  Duration:  {throughput_results['duration_seconds']:.2f} seconds")
    print(f"  RPS:       {throughput_results['requests_per_second']:.2f} requests/second")
    print()
    
    # 6. Concurrent Load Test
    print("=" * 50)
    print("6. Concurrent Load Test (5 workers)")
    print("=" * 50)
    concurrent_results = measure_concurrent_load("/predict", standard_payload, n_workers=5, n_requests_per_worker=20)
    results["benchmarks"]["concurrent_load"] = concurrent_results
    
    if "error" not in concurrent_results:
        print(f"  Workers:      {concurrent_results['workers']}")
        print(f"  Total Reqs:   {concurrent_results['total_requests']}")
        print(f"  Completed:    {concurrent_results['completed']}")
        print(f"  Errors:       {concurrent_results['errors']}")
        print(f"  Mean Latency: {concurrent_results['mean_latency_ms']:.2f} ms")
        print(f"  P95 Latency:  {concurrent_results['p95_latency_ms']:.2f} ms")
        print(f"  RPS:          {concurrent_results['requests_per_second']:.2f}")
    print()
    
    # Summary
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    all_pass = True
    
    if "error" not in predict_results:
        predict_pass = predict_results['p95'] < LATENCY_THRESHOLD_MS
        print(f"  Predict P95 Latency: {predict_results['p95']:.2f}ms {'✅' if predict_pass else '❌'}")
        all_pass = all_pass and predict_pass
    
    if "error" not in health_results:
        health_pass = health_results['p95'] < 100
        print(f"  Health P95 Latency:  {health_results['p95']:.2f}ms {'✅' if health_pass else '❌'}")
        all_pass = all_pass and health_pass
    
    print(f"\nOverall: {'✅ ALL BENCHMARKS PASSED' if all_pass else '❌ SOME BENCHMARKS FAILED'}")
    
    # Save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"performance_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_pass


if __name__ == "__main__":
    success = run_benchmarks()
    sys.exit(0 if success else 1)
