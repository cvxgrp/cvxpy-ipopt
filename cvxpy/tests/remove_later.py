"""
Benchmark for GitHub issue #2205: slow compilation with kron.
Compares CPP, SCIPY, DIFFENGINE backends on the MATLAB formulation.
Tests both real and complex variants. Measures time + peak memory.
Each (config, backend) runs in a subprocess for OOM/crash isolation.

Usage:
  python remove_later.py CPP          # run benchmarks for CPP backend only
  python remove_later.py DIFFENGINE   # run benchmarks for DIFFENGINE backend only
  python remove_later.py --report     # print combined results from all runs
  python remove_later.py CPP --verbose # run with per-reduction timing breakdown

Complex2Real analysis:
  For complex varx, Complex2Real splits kron(I, diag(varx)) into TWO kron ops:
    kron(I, diag(varx_real)) and kron(I, diag(varx_imag))
  Downstream matmuls also split (H' @ kron @ c becomes 4 matmul chains).
  This causes ~4x the work in ConeMatrixStuffing. The split is mathematically
  necessary — the real win is making each kron operation faster (DIFFENGINE).
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CONFIGS = [
    (8, 1, 10),
    (8, 2, 10),
    (16, 2, 10),
    (16, 4, 10),
    (16, 6, 10),
    #(16, 8, 10),
    #(16, 12, 10),
]

BACKENDS = ["CPP", "SCIPY", "COO", "DIFFENGINE"]
TIMEOUT = 120  # seconds per run
RESULTS_FILE = Path(__file__).parent / "benchmark_results.json"


# Worker script executed in subprocess
WORKER_SCRIPT = r'''
import contextlib, io, json, logging, os, sys, time, tracemalloc
import numpy as np

# Suppress sparsediffpy banner (printed to stdout on first use)
_real_stdout = sys.stdout
sys.stdout = io.StringIO()
import cvxpy as cp
import cvxpy.settings as s
sys.stdout = _real_stdout

s.LOGGER.setLevel(logging.ERROR)

N_r, N_t, N_s = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
backend = sys.argv[4]
use_complex = sys.argv[5] == "complex"
verbose = sys.argv[6] == "verbose" if len(sys.argv) > 6 else False

if verbose:
    s.LOGGER.setLevel(logging.INFO)
    s.LOGGER.addHandler(logging.StreamHandler(sys.stderr))

np.random.seed(42)
dim = N_s * N_r * N_t
N_s_N_t = N_s * N_t

if use_complex:
    H = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim))
    x_data = np.random.randint(0, 2, dim).astype(float)
    err = np.random.randn(N_r) + 1j * np.random.randn(N_r)
else:
    H = np.random.randn(dim, dim)
    x_data = np.random.randn(dim)
    err = np.random.randn(N_r)

Err_true = np.kron(np.eye(N_s_N_t), np.diag(err))
y = H.conj().T @ Err_true @ H @ x_data

varx = cp.Variable(N_r, complex=use_complex)
Err_est = cp.kron(np.eye(N_s_N_t), cp.diag(varx))
prob = cp.Problem(cp.Minimize(cp.norm(H.conj().T @ Err_est @ (H @ x_data) - y)))

tracemalloc.start()
t1 = time.perf_counter()
prob.get_problem_data(cp.CLARABEL, canon_backend=backend, verbose=verbose)
elapsed = time.perf_counter() - t1
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

result = {"time": elapsed, "peak_mb": peak / (1024 * 1024)}
# Write result JSON to stderr (stdout may have C-level banner noise)
sys.stderr.write("RESULT:" + json.dumps(result) + "\n")
sys.stderr.flush()
'''


def run_one(N_r, N_t, N_s, backend, use_complex, verbose=False):
    """Run a single benchmark in a subprocess. Returns (time, peak_mb) or error string."""
    ctype = "complex" if use_complex else "real"
    vflag = "verbose" if verbose else "quiet"
    cmd = [
        sys.executable, "-c", WORKER_SCRIPT,
        str(N_r), str(N_t), str(N_s), backend, ctype, vflag,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT
        )
        stderr = result.stderr
        if result.returncode != 0:
            if "MemoryError" in stderr or "Cannot allocate" in stderr:
                return "OOM", None, stderr
            return f"CRASH(rc={result.returncode})", None, stderr
        # Parse RESULT: line from stderr (stdout has C-level banner noise)
        for line in stderr.split("\n"):
            if line.startswith("RESULT:"):
                data = json.loads(line[7:])
                return data["time"], data["peak_mb"], stderr
        return "NO_RESULT", None, stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None, ""
    except Exception as e:
        return f"ERR:{e}", None, ""


def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return []


def save_results(all_results):
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))


def format_result(t, mem):
    if isinstance(t, str):
        return f"{t:>10}", f"{'N/A':>9}"
    return f"{t:>9.3f}s", f"{mem:>8.1f}M"


def print_header():
    print("Issue #2205 Benchmark: kron(I, diag(varx)) MATLAB formulation")
    print("=" * 90)
    header = (
        f"{'type':>7} {'N_r':>4} {'N_t':>4} {'N_s':>4} | {'dim':>5} |"
        f" {'backend':>11} | {'time':>10} {'peak_MB':>9}"
    )
    print(header)
    print("-" * len(header))


def print_report():
    """Print combined results from all saved runs, showing most recent per combination."""
    all_results = load_results()
    if not all_results:
        print("No results found. Run benchmarks first:")
        print(f"  python {__file__} CPP")
        return

    # Keep most recent result per (type, N_r, N_t, N_s, backend)
    latest = {}
    for r in all_results:
        key = (r["type"], r["N_r"], r["N_t"], r["N_s"], r["backend"])
        latest[key] = r

    print_header()

    for use_complex in [False, True]:
        ctype = "complex" if use_complex else "real"
        for N_r, N_t, N_s in CONFIGS:
            dim = N_s * N_r * N_t
            for backend in BACKENDS:
                key = (ctype, N_r, N_t, N_s, backend)
                if key in latest:
                    r = latest[key]
                    t_str, m_str = format_result(r["time"], r["peak_mb"])
                    ts = r.get("timestamp", "")
                    print(
                        f"{ctype:>7} {N_r:>4} {N_t:>4} {N_s:>4} | {dim:>5} |"
                        f" {backend:>11} | {t_str} {m_str}  [{ts}]",
                        flush=True,
                    )
                else:
                    print(
                        f"{ctype:>7} {N_r:>4} {N_t:>4} {N_s:>4} | {dim:>5} |"
                        f" {backend:>11} | {'(no data)':>10} {'':>9}",
                        flush=True,
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark kron backends one at a time. Results accumulate in JSON."
    )
    parser.add_argument(
        "backend", nargs="?", choices=BACKENDS,
        help="Backend to benchmark (one of: CPP, SCIPY, DIFFENGINE)"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print combined results table from all previous runs"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-reduction timing breakdown"
    )
    args = parser.parse_args()

    if args.report:
        print_report()
        return

    if args.backend is None:
        parser.error(
            "backend is required (one of: CPP, SCIPY, DIFFENGINE). "
            "Use --report to view results."
        )

    backend = args.backend
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_results = load_results()

    print_header()

    for use_complex in [False, True]:
        ctype = "complex" if use_complex else "real"
        for N_r, N_t, N_s in CONFIGS:
            dim = N_s * N_r * N_t
            t, mem, stderr = run_one(N_r, N_t, N_s, backend, use_complex)

            t_str, m_str = format_result(t, mem)
            print(
                f"{ctype:>7} {N_r:>4} {N_t:>4} {N_s:>4} | {dim:>5} |"
                f" {backend:>11} | {t_str} {m_str}",
                flush=True,
            )

            all_results.append({
                "timestamp": timestamp,
                "type": ctype, "N_r": N_r, "N_t": N_t, "N_s": N_s,
                "dim": dim, "backend": backend, "time": t, "peak_mb": mem,
            })
            # Save after each result so partial runs aren't lost
            save_results(all_results)

    # Verbose per-reduction breakdown
    if args.verbose:
        print(f"\n{'=' * 90}")
        print(f"Per-reduction timing for (N_r=16, N_t=2, N_s=10) — real vs complex, {backend} backend")
        print("=" * 90)
        for use_complex in [False, True]:
            ctype = "complex" if use_complex else "real"
            print(f"\n--- {ctype} ---")
            _, _, stderr = run_one(16, 2, 10, backend, use_complex, verbose=True)
            for line in stderr.split("\n"):
                if "took" in line or "Reduction chain" in line or "Finished" in line:
                    cleaned = line.split(") ", 1)[-1] if ") " in line else line
                    print(f"  {cleaned.strip()}")

    print(f"\nResults saved to {RESULTS_FILE}")
    print(
        f"Run 'python {__file__} --report' to see combined results."
    )


if __name__ == "__main__":
    main()
