
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {"passage_id", "text"}


def load_passages(csv_path: str | Path) -> list[dict[str, Any]]:
    """
    Load passages from a CSV file and validate their structure.

    Args:
        csv_path: Path to the CSV dataset.

    Returns:
        A list of passage dictionaries.
    """

    file_path = Path(csv_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    dataframe = pd.read_csv(file_path)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("Dataset contains no passages.")

    if dataframe["passage_id"].isna().any():
        raise ValueError("Dataset contains missing passage IDs.")

    if dataframe["passage_id"].duplicated().any():
        raise ValueError("Dataset contains duplicate passage IDs.")

    if dataframe["text"].isna().any():
        raise ValueError("Dataset contains missing passage text.")

    if dataframe["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Dataset contains empty passage text.")

    passages = dataframe.to_dict(orient="records")

    print(f"Successfully loaded {len(passages)} passages.")
    print(f"Dataset: {file_path.name}")

    return passages


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "data"
        / "synthetic"
        / "synthetic_benign_passages.csv"
    )

    documents = load_passages(dataset_path)

    print("\nFirst passage:")
    print(documents[0])
