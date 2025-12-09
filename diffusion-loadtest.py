#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import httpx


@dataclass
class Result:
    idx: int
    ok: bool
    status_code: Optional[int]
    elapsed_s: float
    error: Optional[str]


def percentile(sorted_vals: List[float], p: float) -> float:
    """Nearest-rank percentile. Input must be sorted."""
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = int((p / 100.0) * (len(sorted_vals) - 1))
    return sorted_vals[k]


async def worker(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    idx: int,
    out_dir: Optional[str],
) -> Result:
    async with sem:
        t0 = time.perf_counter()
        status_code = None
        try:
            resp = await client.post(url, json=payload)
            status_code = resp.status_code
            elapsed = time.perf_counter() - t0

            if resp.status_code < 200 or resp.status_code >= 300:
                # try to include response body snippet for debugging
                body_snip = resp.text[:300].replace("\n", " ")
                return Result(idx, False, status_code, elapsed, f"HTTP {status_code}: {body_snip}")

            data = resp.json()

            # Validate expected schema: data[0].b64_json
            b64 = None
            try:
                b64 = data["data"][0]["b64_json"]
            except Exception:
                # keep the first ~300 chars of JSON to help debugging
                snip = json.dumps(data)[:300]
                return Result(idx, False, status_code, elapsed, f"Unexpected JSON schema: {snip}")

            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                img_bytes = base64.b64decode(b64)
                path = os.path.join(out_dir, f"image_{idx:05d}.png")
                with open(path, "wb") as f:
                    f.write(img_bytes)

            return Result(idx, True, status_code, elapsed, None)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            return Result(idx, False, status_code, elapsed, f"{type(e).__name__}: {e}")


async def run(args: argparse.Namespace) -> int:
    url = args.endpoint.rstrip("/") + "/v1/images/generations"

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "n": args.n,
        "size": args.size,
        "response_format": "b64_json",
    }

    sem = asyncio.Semaphore(args.concurrency)

    timeout = httpx.Timeout(connect=args.connect_timeout, read=args.read_timeout, write=args.write_timeout, pool=args.pool_timeout)
    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency)

    headers = {"Content-Type": "application/json"}
    if args.bearer_token:
        headers["Authorization"] = f"Bearer {args.bearer_token}"

    async with httpx.AsyncClient(timeout=timeout, limits=limits, headers=headers, verify=not args.insecure) as client:
        tasks = [
            asyncio.create_task(worker(sem, client, url, payload, i, args.output_dir))
            for i in range(1, args.requests + 1)
        ]
        results = await asyncio.gather(*tasks)

    # Sort by idx for stable per-request output
    results.sort(key=lambda r: r.idx)

    # Print per-request results
    print("idx\tok\tstatus\tms\terror")
    for r in results:
        ms = r.elapsed_s * 1000.0
        err = r.error.replace("\t", " ") if r.error else ""
        status = r.status_code if r.status_code is not None else "-"
        print(f"{r.idx}\t{int(r.ok)}\t{status}\t{ms:.2f}\t{err}")

    # Summary
    times_ok = sorted([r.elapsed_s for r in results if r.ok])
    times_all = sorted([r.elapsed_s for r in results])

    ok_count = sum(1 for r in results if r.ok)
    err_count = len(results) - ok_count

    def fmt_ms(x: float) -> str:
        return "nan" if x != x else f"{x*1000.0:.2f}"

    avg_ok = statistics.mean(times_ok) if times_ok else float("nan")
    avg_all = statistics.mean(times_all) if times_all else float("nan")

    print("\nSummary")
    print(f"Requests:     {len(results)}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Success:      {ok_count}")
    print(f"Errors:       {err_count}")
    print(f"Avg (ok):     {fmt_ms(avg_ok)} ms")
    print(f"Avg (all):    {fmt_ms(avg_all)} ms")
    print(f"Min (ok):     {fmt_ms(times_ok[0])} ms" if times_ok else "Min (ok):     nan")
    print(f"Max (ok):     {fmt_ms(times_ok[-1])} ms" if times_ok else "Max (ok):     nan")
    if times_ok:
        print(f"p50 (ok):     {fmt_ms(percentile(times_ok, 50))} ms")
        print(f"p95 (ok):     {fmt_ms(percentile(times_ok, 95))} ms")
        print(f"p99 (ok):     {fmt_ms(percentile(times_ok, 99))} ms")

    # Non-zero exit if any errors (useful for CI)
    return 0 if err_count == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Concurrent load test for /v1/images/generations (b64_json).")
    ap.add_argument("--endpoint", required=True, help="Base endpoint, e.g. https://...proxy.runpod.net")
    ap.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
    ap.add_argument("--prompt", default="a small robot penguin fixing a linux server rack, photo-realistic, dramatic lighting")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--n", type=int, default=1)

    ap.add_argument("-r", "--requests", type=int, default=50, help="Total number of requests")
    ap.add_argument("-c", "--concurrency", type=int, default=10, help="Number of in-flight requests")

    ap.add_argument("--output-dir", default=None, help="If set, saves images as PNG files here")

    ap.add_argument("--bearer-token", default=None, help="Optional Authorization Bearer token")
    ap.add_argument("--insecure", action="store_true", help="Disable TLS verification")

    # Timeouts (seconds)
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--read-timeout", type=float, default=300.0)
    ap.add_argument("--write-timeout", type=float, default=30.0)
    ap.add_argument("--pool-timeout", type=float, default=30.0)

    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
