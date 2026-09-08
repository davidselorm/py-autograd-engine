import time
from autograd import Value, MLP

def run_benchmark():
    print("[BENCHMARK] Evaluating 3-layer MLP execution speed...")
    model = MLP(3, [16, 16, 1])
    n_iters = 2000
    inputs = [Value(0.5), Value(-0.2), Value(1.1)]

    t0 = time.perf_counter()
    for _ in range(n_iters):
        model.zero_grad()
        out = model(inputs)
        out.backward()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    ops_per_sec = n_iters / elapsed
    print(f"Total passes: {n_iters} forward+backward")
    print(f"Elapsed Time: {elapsed:.4f}s")
    print(f"Throughput:   {ops_per_sec:.2f} iterations/sec")

if __name__ == '__main__':
    run_benchmark()
