
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_PATH = BASE_DIR / "data/real/hotpot_original_passages.csv"
ATTACK_PATH = BASE_DIR / "data/attacks/hotpot_attack_modified_passages.csv"

original = pd.read_csv(ORIGINAL_PATH)
attacks = pd.read_csv(ATTACK_PATH)

print("=" * 70)
print("RAG SECURITY DATASET VALIDATION")
print("=" * 70)

# 1. Check duplicate passage IDs
print("\n1. DUPLICATE ID CHECK")

print("Original duplicates:", original["passage_id"].duplicated().sum())
print("Attack duplicates:", attacks["passage_id"].duplicated().sum())

# 2. Check passage ID alignment
print("\n2. PASSAGE ID ALIGNMENT")

original_ids = set(original["passage_id"])
attack_ids = set(attacks["passage_id"])

print("IDs only in original:", len(original_ids - attack_ids))
print("IDs only in attack dataset:", len(attack_ids - original_ids))
print("Same passage IDs:", original_ids == attack_ids)

# 3. Compare original and attack passage text
print("\n3. TEXT COMPARISON")

merged = original[
    ["passage_id", "text", "is_supporting_document"]
].merge(
    attacks[
        [
            "passage_id",
            "text",
            "is_supporting_document",
            "injection_present",
            "attack_type",
            "injection_text",
        ]
    ],
    on="passage_id",
    how="inner",
    suffixes=("_original", "_attack"),
    validate="one_to_one",
)

merged["text_changed"] = (
    merged["text_original"] != merged["text_attack"]
)

merged["injection_present"] = (
    merged["injection_present"].fillna(0).astype(int)
)

marked = merged["injection_present"] == 1
unmarked = ~marked

print("Matched passages:", len(merged))
print("Text changed:", merged["text_changed"].sum())
print("Text unchanged:", (~merged["text_changed"]).sum())

# 4. Check injection information
print("\n4. INJECTION CHECK")

print("Marked injection passages:", marked.sum())
print("Unmarked passages:", unmarked.sum())

print(
    "Marked passages with changed text:",
    (marked & merged["text_changed"]).sum(),
)

print(
    "Unmarked passages with changed text:",
    (unmarked & merged["text_changed"]).sum(),
)

print(
    "Marked passages without text changes:",
    (marked & ~merged["text_changed"]).sum(),
)

# 5. Verify recorded injection text
injection_text = merged["injection_text"].fillna("").astype(str)
attack_text = merged["text_attack"].fillna("").astype(str)

merged["injection_matches_text"] = [
    bool(injection.strip()) and injection in text
    for injection, text in zip(injection_text, attack_text)
]

print(
    "Recorded injections found in passage text:",
    (marked & merged["injection_matches_text"]).sum(),
)

print(
    "Marked passages with missing/mismatched injection text:",
    (marked & ~merged["injection_matches_text"]).sum(),
)

# 6. Check supporting-document labels
print("\n5. SUPPORTING DOCUMENT CHECK")

supporting_labels_match = (
    merged["is_supporting_document_original"]
    == merged["is_supporting_document_attack"]
)

print("Supporting labels unchanged:", supporting_labels_match.all())
print("Supporting label mismatches:", (~supporting_labels_match).sum())

# 7. Attack type distribution
print("\n6. ATTACK TYPE DISTRIBUTION")

print(
    attacks.loc[
        attacks["injection_present"] == 1, "attack_type"
    ].value_counts(dropna=False).to_string()
)

# 8. Final validation summary
print("\n7. VALIDATION SUMMARY")

checks = {
    "No duplicate original IDs": not original["passage_id"].duplicated().any(),
    "No duplicate attack IDs": not attacks["passage_id"].duplicated().any(),
    "Same passage IDs": original_ids == attack_ids,
    "All marked passages have changed text": (
        (marked & merged["text_changed"]).sum() == marked.sum()
    ),
    "Unmarked passages are unchanged": (
        not (unmarked & merged["text_changed"]).any()
    ),
    "All injection texts match": (
        not (marked & ~merged["injection_matches_text"]).any()
    ),
    "Supporting labels unchanged": supporting_labels_match.all(),
}

for check, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {check}")

print(
    "\nOVERALL RESULT:",
    "PASSED" if all(checks.values()) else "REQUIRES INVESTIGATION"
)

print("\nDATASET VALIDATION COMPLETED")
