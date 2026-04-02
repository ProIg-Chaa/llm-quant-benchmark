#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List


NUMERIC_FIELDS = [
    "ttft_ms",
    "total_latency_ms",
    "decode_tokens_per_s",
    "request_tokens_per_s",
    "peak_gpu_mem_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate benchmark CSV files into a markdown table.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV files.")
    parser.add_argument("--markdown-out", required=True, help="Output markdown table path.")
    parser.add_argument("--title", default="Benchmark Summary", help="Markdown title.")
    return parser.parse_args()


def load_rows(paths: List[Path]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    continue
                grouped[row["quant_method"]].append(row)
    return grouped


def mean_field(rows: List[Dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field)]
    return mean(values) if values else float("nan")


def render_markdown(grouped: Dict[str, List[Dict[str, str]]]) -> str:
    order = ["fp16", "bnb_int8", "bnb_int4", "awq", "gptq"]
    ordered_keys = [key for key in order if key in grouped] + sorted(
        key for key in grouped.keys() if key not in order
    )
    lines = [
        None,
        "",
        "| quant_method | rows | avg_ttft_ms | avg_total_latency_ms | avg_decode_tokens_per_s | avg_request_tokens_per_s | avg_peak_gpu_mem_mb |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for quant_method in ordered_keys:
        rows = grouped[quant_method]
        lines.append(
            "| {quant_method} | {rows_count} | {ttft:.3f} | {total:.3f} | {decode:.3f} | {request:.3f} | {mem:.3f} |".format(
                quant_method=quant_method,
                rows_count=len(rows),
                ttft=mean_field(rows, "ttft_ms"),
                total=mean_field(rows, "total_latency_ms"),
                decode=mean_field(rows, "decode_tokens_per_s"),
                request=mean_field(rows, "request_tokens_per_s"),
                mem=mean_field(rows, "peak_gpu_mem_mb"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_paths = [Path(item) for item in args.inputs]
    markdown_out = Path(args.markdown_out)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)

    grouped = load_rows(input_paths)
    content = render_markdown(grouped)
    content = content.replace("None", f"# {args.title}", 1)
    markdown_out.write_text(content, encoding="utf-8")
    print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
