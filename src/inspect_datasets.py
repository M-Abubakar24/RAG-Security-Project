
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FILES = {
    "Original Passages": BASE_DIR / "data/real/hotpot_original_passages.csv",
    "Attack Passages": BASE_DIR / "data/attacks/hotpot_attack_modified_passages.csv",
    "Questions and Answers": BASE_DIR / "data/real/hotpot_questions_answers.csv",
}

for name, path in FILES.items():
    print("\n" + "=" * 65)
    print(f"DATASET: {name}")
    print("=" * 65)

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        continue

    try:
        df = pd.read_csv(path)

        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes)

        print("\nFirst 2 records:")
        print(df.head(2).to_string(index=False))

        print("\nMissing values:")
        print(df.isnull().sum())

    except Exception as error:
        print(f"Error reading dataset: {error}")

print("\nDATASET INSPECTION COMPLETED")
