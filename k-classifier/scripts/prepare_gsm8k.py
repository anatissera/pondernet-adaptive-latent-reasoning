import argparse
import json
import re
from datasets import load_dataset


def extract_final_answer(answer: str) -> str:
    # GSM8K suele traer: "... explicación ... #### 72"
    if "####" in answer:
        final = answer.split("####")[-1].strip()
    else:
        final = answer.strip()

    # Limpieza simple para números con comas, unidades, etc.
    final = final.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", final)
    return match.group(0) if match else final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--output", default="k-classifier/data/gsm8k_test_100.jsonl")
    args = parser.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split=args.split)

    if args.n_examples is not None:
        ds = ds.select(range(min(args.n_examples, len(ds))))

    with open(args.output, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            row = {
                "id": f"gsm8k_{args.split}_{i:05d}",
                "input": ex["question"],
                "gold": extract_final_answer(ex["answer"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} examples to {args.output}")


if __name__ == "__main__":
    main()
