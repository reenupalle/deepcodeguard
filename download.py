"""Download the Devign / CodeXGLUE defect-detection dataset to data/raw/."""
from collections import Counter
from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path("data/raw")
EXPECTED = {"train": 21854, "validation": 2732, "test": 2732}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("google/code_x_glue_cc_defect_detection")

    for split, expected in EXPECTED.items():
        data = ds[split]
        out = RAW_DIR / f"{split}.jsonl"
        data.to_json(out, lines=True)

        labels = Counter(int(t) for t in data["target"])
        projects = Counter(data["project"])
        status = "OK" if len(data) == expected else f"MISMATCH (expected {expected})"
        print(f"{split:<11} {len(data):>6} rows  {status}")
        print(f"            labels   {dict(labels)}")
        print(f"            projects {dict(projects)}")

    print("columns:", ds["train"].column_names)


if __name__ == "__main__":
    main()