import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from benchmark.concurrent_client import run_concurrent
from benchmark.naive_client import run_naive


def build_prompt(target_tokens: int) -> str:
    seed = "Explain dynamic batching tradeoffs in GPU inference systems."
    return " ".join([seed] * max(target_tokens // 10, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM inference benchmark")
    parser.add_argument("--mode", choices=["naive", "concurrent", "both"], default="both")
    parser.add_argument("--num-requests", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--prompt-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--auth-token", default="your-secret-token-here")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--output", default="benchmark/results/run_latest.json")
    parser.add_argument(
        "--log-node",
        action="store_true",
        help="Record X-Routed-Node response header and print per-node counts",
    )
    return parser.parse_args()


def _node_counts(results: list) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in results:
        if r.node:
            c[r.node] += 1
        else:
            c["unknown"] += 1
    return dict(c)


def summarize_naive(results: list) -> dict:
    total_time = sum(r.latency_ms for r in results) / 1000
    completion_tokens = sum(r.completion_tokens for r in results)
    return {
        "total_time_seconds": total_time,
        "requests_per_second": len(results) / max(total_time, 1e-9),
        "tokens_per_second": completion_tokens / max(total_time, 1e-9),
    }


async def main() -> None:
    args = parse_args()
    prompt = build_prompt(args.prompt_tokens)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timeout = httpx.Timeout(180.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        data: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "num_requests": args.num_requests,
                "concurrency": args.concurrency,
                "prompt_tokens": args.prompt_tokens,
                "max_tokens": args.max_tokens,
                "base_url": args.base_url,
                "model": args.model,
            },
            "runs": {},
        }

        if args.mode in {"naive", "both"}:
            naive = await run_naive(
                client=client,
                base_url=args.base_url,
                token=args.auth_token,
                num_requests=args.num_requests,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                capture_node=args.log_node,
            )
            data["runs"]["naive"] = summarize_naive(naive)
            if args.log_node:
                data["runs"]["naive_node_counts"] = _node_counts(naive)

        if args.mode in {"concurrent", "both"}:
            concurrent_results, concurrent = await run_concurrent(
                client=client,
                base_url=args.base_url,
                token=args.auth_token,
                num_requests=args.num_requests,
                concurrency=args.concurrency,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                capture_node=args.log_node,
            )
            data["runs"]["concurrent"] = asdict(concurrent)
            if args.log_node:
                data["runs"]["concurrent_node_counts"] = _node_counts(concurrent_results)

        naive_tps = data["runs"].get("naive", {}).get("tokens_per_second", 0.0)
        concurrent_tps = data["runs"].get("concurrent", {}).get("tokens_per_second", 0.0)
        data["throughput_multiplier"] = (concurrent_tps / naive_tps) if naive_tps else None

        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved results to {output_path}")
        if "naive" in data["runs"]:
            print(f"Naive tok/s:      {data['runs']['naive']['tokens_per_second']:.2f}")
        if "concurrent" in data["runs"]:
            print(f"Concurrent tok/s: {data['runs']['concurrent']['tokens_per_second']:.2f}")
        if data["throughput_multiplier"] is not None:
            print(f"Multiplier:       {data['throughput_multiplier']:.2f}x")
        if args.log_node:
            if "naive_node_counts" in data["runs"]:
                print(f"Naive node counts:      {data['runs']['naive_node_counts']}")
            if "concurrent_node_counts" in data["runs"]:
                print(f"Concurrent node counts: {data['runs']['concurrent_node_counts']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

