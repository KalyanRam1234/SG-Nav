#!/usr/bin/env python3
"""Test parallel Ollama VLM/LLM calls to measure throughput gains."""

import time
import io
import base64
import ollama
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = "llama3.2-vision:latest"

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Create a dummy test image (returned as base64)
def make_test_image():
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return pil_to_base64(Image.fromarray(arr))

# Single VLM call
def vlm_call(idx, image):
    prompt = f"Describe the spatial relationship between object A and object B in this image. Reply in 5 words or less."
    t0 = time.time()
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt, "images": [image]}],
        keep_alive=-1,
    )
    elapsed = time.time() - t0
    text = resp["message"]["content"].strip()
    return idx, elapsed, text

# Single text-only LLM call
def llm_call(idx):
    prompt = f"What is the spatial relationship between a chair and a table? Reply in 5 words or less."
    t0 = time.time()
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        keep_alive=-1,
    )
    elapsed = time.time() - t0
    text = resp["message"]["content"].strip()
    return idx, elapsed, text


def benchmark(label, fn, n_calls, max_workers):
    """Run fn n_calls times with given parallelism."""
    print(f"\n{'='*60}")
    print(f"  {label}: {n_calls} calls, max_workers={max_workers}")
    print(f"{'='*60}")
    
    t_start = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fn, i) for i in range(n_calls)]
        for future in as_completed(futures):
            results.append(future.result())
    
    t_total = time.time() - t_start
    results.sort(key=lambda x: x[0])
    
    times = [r[1] for r in results]
    print(f"  Per-call times: {['%.2fs' % t for t in times]}")
    print(f"  Avg per call:   {np.mean(times):.2f}s")
    print(f"  Total wall time: {t_total:.2f}s")
    print(f"  Throughput:      {n_calls / t_total:.2f} calls/sec")
    return t_total


if __name__ == "__main__":
    N = 8  # number of calls to test
    img = make_test_image()

    # Warm up
    print("Warming up model...")
    ollama.chat(model=MODEL, messages=[{"role": "user", "content": "hi"}], keep_alive=-1)
    print("Model warm.\n")

    # --- VLM benchmarks ---
    vlm_fn = lambda i: vlm_call(i, img)

    t_serial = benchmark("VLM SERIAL", vlm_fn, N, max_workers=1)
    t_par2   = benchmark("VLM PARALLEL x2", vlm_fn, N, max_workers=2)
    t_par4   = benchmark("VLM PARALLEL x4", vlm_fn, N, max_workers=4)
    t_par8   = benchmark("VLM PARALLEL x8", vlm_fn, N, max_workers=8)

    print(f"\n{'='*60}")
    print(f"  VLM SUMMARY ({N} calls)")
    print(f"  Serial:     {t_serial:.2f}s")
    print(f"  Parallel x2: {t_par2:.2f}s  (speedup: {t_serial/t_par2:.1f}x)")
    print(f"  Parallel x4: {t_par4:.2f}s  (speedup: {t_serial/t_par4:.1f}x)")
    print(f"  Parallel x8: {t_par8:.2f}s  (speedup: {t_serial/t_par8:.1f}x)")
    print(f"{'='*60}")

    # --- Text LLM benchmarks ---
    t_serial = benchmark("LLM SERIAL", llm_call, N, max_workers=1)
    t_par4   = benchmark("LLM PARALLEL x4", llm_call, N, max_workers=4)

    print(f"\n{'='*60}")
    print(f"  LLM SUMMARY ({N} calls)")
    print(f"  Serial:     {t_serial:.2f}s")
    print(f"  Parallel x4: {t_par4:.2f}s  (speedup: {t_serial/t_par4:.1f}x)")
    print(f"{'='*60}")
