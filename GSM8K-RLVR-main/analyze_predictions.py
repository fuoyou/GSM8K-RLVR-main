import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GSM8K eval predictions.")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sample_limit", type=int, default=10)
    return parser.parse_args()


def short_text(text: str, n: int = 220) -> str:
    return " ".join((text or "").split())[:n]


def main():
    args = parse_args()
    pred_path = Path(args.predictions)
    rows = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    format_pat = re.compile(r"####.*?([\-]?[\d,]+(?:\.\d+)?)")
    next_question_pat = re.compile(r"\bQuestion:\s", re.IGNORECASE)

    total = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    wrong = total - correct

    has_format = []
    pred_null = []
    next_question = []
    long_output = []
    format_miss_wrong = []
    numeric_wrong = []

    for r in rows:
        text = r.get("pred_text") or ""
        if format_pat.search(text):
            has_format.append(r)
        if r.get("pred_value") is None:
            pred_null.append(r)
        if next_question_pat.search(text):
            next_question.append(r)
        if len(text) >= 500:
            long_output.append(r)

        if not r.get("correct"):
            if r.get("pred_value") is None:
                format_miss_wrong.append(r)
            else:
                numeric_wrong.append(r)

    lines = []
    lines.append(f"predictions={pred_path}")
    lines.append(f"total={total}")
    lines.append(f"correct={correct}")
    lines.append(f"wrong={wrong}")
    lines.append(f"accuracy={100.0 * correct / total if total else 0.0:.2f}%")
    lines.append("")
    lines.append(f"has_hash_number={len(has_format)} ({100.0 * len(has_format) / total if total else 0.0:.2f}%)")
    lines.append(f"pred_value_null={len(pred_null)} ({100.0 * len(pred_null) / total if total else 0.0:.2f}%)")
    lines.append(f"contains_next_question={len(next_question)} ({100.0 * len(next_question) / total if total else 0.0:.2f}%)")
    lines.append(f"long_output_len_ge_500={len(long_output)} ({100.0 * len(long_output) / total if total else 0.0:.2f}%)")
    lines.append("")
    lines.append(
        f"wrong_bucket_format_miss={len(format_miss_wrong)} ({100.0 * len(format_miss_wrong) / wrong if wrong else 0.0:.2f}% of wrong)"
    )
    lines.append(
        f"wrong_bucket_numeric_wrong={len(numeric_wrong)} ({100.0 * len(numeric_wrong) / wrong if wrong else 0.0:.2f}% of wrong)"
    )
    lines.append("")

    lines.append("SAMPLES format_miss")
    for r in format_miss_wrong[: args.sample_limit]:
        lines.append(
            f"idx={r.get('idx')} gt={r.get('gt_value')} pred={r.get('pred_value')} text={short_text(r.get('pred_text') or '')}"
        )

    lines.append("")
    lines.append("SAMPLES numeric_wrong")
    for r in numeric_wrong[: args.sample_limit]:
        lines.append(
            f"idx={r.get('idx')} gt={r.get('gt_value')} pred={r.get('pred_value')} text={short_text(r.get('pred_text') or '')}"
        )

    lines.append("")
    lines.append("SAMPLES contains_next_question")
    for r in next_question[: args.sample_limit]:
        lines.append(
            f"idx={r.get('idx')} gt={r.get('gt_value')} pred={r.get('pred_value')} correct={r.get('correct')} text={short_text(r.get('pred_text') or '')}"
        )

    report = "\n".join(lines) + "\n"
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"saved_report={out_path}")


if __name__ == "__main__":
    main()
