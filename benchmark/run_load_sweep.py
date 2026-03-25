import argparse
import csv
import json
from pathlib import Path

import httpx

from benchmark.concurrent_client import run_concurrent
from benchmark.run_benchmark import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run concurrency load sweep")
    parser.add_argument("--num-requests", type=int, default=60)
    parser.add_argument("--prompt-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--auth-token", default="your-secret-token-here")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--output-json", default="benchmark/results/load_sweep.json")
    parser.add_argument("--output-csv", default="benchmark/results/load_sweep.csv")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    prompt = build_prompt(args.prompt_tokens)
    concurrencies = [1, 2, 4, 8, 12, 16]
    rows: list[dict] = []
    timeout = httpx.Timeout(180.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        for c in concurrencies:
            _, metrics = await run_concurrent(
                client=client,
                base_url=args.base_url,
                token=args.auth_token,
                num_requests=args.num_requests,
                concurrency=c,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )
            rows.append(
                {
                    "concurrency": c,
                    "tokens_per_second": metrics.tokens_per_second,
                    "latency_p95_ms": metrics.latency_p95_ms,
                    "error_rate": metrics.error_rate,
                    "gpu_util_avg": "",
                    "vram_used_mb_avg": "",
                }
            )
            print(
                f"c={c:<2} tok/s={metrics.tokens_per_second:.2f} "
                f"p95={metrics.latency_p95_ms:.2f}ms err={metrics.error_rate:.2f}%"
            )

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "concurrency",
                "tokens_per_second",
                "latency_p95_ms",
                "error_rate",
                "gpu_util_avg",
                "vram_used_mb_avg",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved sweep JSON: {json_path}")
    print(f"Saved sweep CSV:  {csv_path}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

