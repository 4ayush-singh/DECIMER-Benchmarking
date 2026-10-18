#!/usr/bin/env python3
"""
Evaluate DECIMER-extracted SMILES against a ground-truth CSV.

Expected DECIMER input (from run_extraction.py): output/results.csv
Expected ground truth columns: paperID, name of molecule, canonical_SMILES

Outputs in evaluation_output/:
  prediction_matches.csv  one row per extracted structure, with its best GT match
  ground_truth_coverage.csv one row per GT molecule, with its best extracted match
  summary.json            overall and per-paper metrics

Install dependencies:
  pip install pandas rdkit

Run:
  python evaluate_smiles.py

IMPORTANT:
  Set PAPER_ID_MAP below if PDF filenames do not contain their paperID.
  Example: {"paper_1.pdf": 1, "paper_2.pdf": 2}
"""

from pathlib import Path
import json
import re

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.warning")

# -------- CONFIGURATION --------
PREDICTIONS_CSV = Path("output/results.csv")
GROUND_TRUTH_CSV = Path(
    "ground_truth_SMILES_confirmed_1,2,4,6,16,19.csv"
)
OUTPUT_DIR = Path("evaluation_output")

# Map source PDF filename (or relative path as shown in results.csv) to paperID.
# This is the most reliable way to restrict comparisons to the right paper.
# Example: {"paper1.pdf": 1, "paper2.pdf": 2}
PAPER_ID_MAP = {}

# If no explicit map entry matches, try to infer paperID from the filename
# using a pattern like "paper_19.pdf", "19.pdf", or "paper19.pdf".
# If inference fails, compare against all ground-truth molecules.
TANIMOTO_THRESHOLD = 0.50
MORGAN_RADIUS = 2
MORGAN_FP_SIZE = 2048


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_molecule(smiles):
    """Return (RDKit molecule, canonical isomeric SMILES, InChIKey)."""
    smiles = clean_text(smiles)
    if not smiles:
        return None, None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = None
    return mol, canonical, inchikey


def infer_paper_id(source_pdf):
    source = clean_text(source_pdf).replace("\\", "/")
    if not source:
        return None

    # First allow an explicit map by full path or basename.
    basename = source.rsplit("/", 1)[-1]
    for key, value in PAPER_ID_MAP.items():
        if source == key.replace("\\", "/") or basename == key.replace("\\", "/").rsplit("/", 1)[-1]:
            return str(value)

    # Common filename patterns: paper_19.pdf, paper19.pdf, 19.pdf.
    stem = basename.rsplit(".", 1)[0]
    match = re.search(r"(?:paper|study|document)?[_\-\s]*(\d+)$", stem, re.I)
    if match:
        return match.group(1)
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {PREDICTIONS_CSV}. Run DECIMER extraction first "
            "or update PREDICTIONS_CSV."
        )
    if not GROUND_TRUTH_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {GROUND_TRUTH_CSV}. Update GROUND_TRUTH_CSV "
            "to the path of your ground-truth CSV."
        )

    pred_df = pd.read_csv(PREDICTIONS_CSV)
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)

    required_pred = {"predicted_smiles"}
    required_gt = {"paperID", "name of molecule", "canonical_SMILES"}
    if not required_pred.issubset(pred_df.columns):
        raise ValueError(f"Prediction CSV needs columns: {sorted(required_pred)}")
    if not required_gt.issubset(gt_df.columns):
        raise ValueError(f"Ground-truth CSV needs columns: {sorted(required_gt)}")

    # Prepare ground-truth molecules and fingerprints.
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_FP_SIZE
    )
    gt_records = []
    for idx, row in gt_df.iterrows():
        smiles = clean_text(row["canonical_SMILES"])
        mol, canonical, inchikey = parse_molecule(smiles)
        gt_records.append({
            "gt_index": int(idx),
            "paperID": clean_text(row["paperID"]),
            "molecule_name": clean_text(row["name of molecule"]),
            "ground_truth_smiles": smiles,
            "gt_canonical_smiles": canonical,
            "gt_inchikey": inchikey,
            "gt_valid": mol is not None,
            "mol": mol,
            "fp": fp_generator.GetFingerprint(mol) if mol is not None else None,
        })

    valid_gt = [r for r in gt_records if r["mol"] is not None]
    if not valid_gt:
        raise ValueError("No valid ground-truth SMILES could be parsed by RDKit.")

    prediction_rows = []
    best_by_gt = {}

    for pred_idx, row in pred_df.iterrows():
        pred_smiles = clean_text(row.get("predicted_smiles", ""))
        pred_mol, pred_canonical, pred_inchikey = parse_molecule(pred_smiles)
        pred_fp = fp_generator.GetFingerprint(pred_mol) if pred_mol is not None else None

        source_pdf = clean_text(row.get("source_pdf", ""))
        inferred_paper = infer_paper_id(source_pdf)

        # Restrict to a paper's GT rows when paperID can be identified.
        candidates = [g for g in valid_gt if inferred_paper is not None and g["paperID"] == inferred_paper]
        comparison_scope = "same_paper"
        if not candidates:
            candidates = valid_gt
            comparison_scope = "all_papers"

        scored = []
        if pred_mol is not None:
            for g in candidates:
                similarity = DataStructs.TanimotoSimilarity(pred_fp, g["fp"])
                exact = pred_smiles == g["ground_truth_smiles"]
                canonical_match = pred_canonical == g["gt_canonical_smiles"]
                inchikey_match = bool(pred_inchikey and g["gt_inchikey"] and pred_inchikey == g["gt_inchikey"])
                scored.append((similarity, exact, canonical_match, inchikey_match, g))

        scored.sort(key=lambda x: (x[2], x[3], x[0]), reverse=True)
        if scored:
            similarity, exact, canonical_match, inchikey_match, best = scored[0]
            matched_name = best["molecule_name"]
            matched_paper = best["paperID"]
            matched_gt_index = best["gt_index"]
            best_similarity = float(similarity)
            exact_match = bool(exact)
            canonical_match = bool(canonical_match)
            inchikey_match = bool(inchikey_match)
            threshold_match = best_similarity >= TANIMOTO_THRESHOLD
            # Coverage records the best prediction for each GT, even if several
            # predictions have the same best GT. Coverage is molecule-level.
            old = best_by_gt.get(best["gt_index"])
            candidate_quality = (canonical_match, inchikey_match, best_similarity)
            if old is None or candidate_quality > old["quality"]:
                best_by_gt[best["gt_index"]] = {
                    "pred_index": int(pred_idx),
                    "structure_id": clean_text(row.get("structure_id", "")),
                    "predicted_smiles": pred_smiles,
                    "similarity": best_similarity,
                    "canonical_match": canonical_match,
                    "inchikey_match": inchikey_match,
                    "quality": candidate_quality,
                }
        else:
            matched_name = matched_paper = matched_gt_index = None
            best_similarity = None
            exact_match = canonical_match = inchikey_match = threshold_match = False

        prediction_rows.append({
            "prediction_row_index": int(pred_idx),
            "structure_id": clean_text(row.get("structure_id", "")),
            "source_pdf": source_pdf,
            "inferred_paperID": inferred_paper,
            "page_number": row.get("page_number", None),
            "image_path": clean_text(row.get("image_path", "")),
            "predicted_smiles": pred_smiles or None,
            "prediction_valid": pred_mol is not None,
            "predicted_canonical_smiles": pred_canonical,
            "predicted_inchikey": pred_inchikey,
            "comparison_scope": comparison_scope,
            "best_match_paperID": matched_paper,
            "best_match_molecule_name": matched_name,
            "best_match_ground_truth_index": matched_gt_index,
            "exact_string_match": exact_match,
            "canonical_smiles_match": canonical_match,
            "inchikey_match": inchikey_match,
            "best_tanimoto_similarity": best_similarity,
            "tanimoto_at_or_above_threshold": threshold_match,
            "tanimoto_threshold": TANIMOTO_THRESHOLD,
        })

    matches_df = pd.DataFrame(prediction_rows)
    matches_path = OUTPUT_DIR / "prediction_matches.csv"
    matches_df.to_csv(matches_path, index=False)

    coverage_rows = []
    for g in gt_records:
        best = best_by_gt.get(g["gt_index"])
        coverage_rows.append({
            "paperID": g["paperID"],
            "molecule_name": g["molecule_name"],
            "ground_truth_smiles": g["ground_truth_smiles"],
            "ground_truth_valid": g["gt_valid"],
            "matched_by_prediction": best is not None,
            "best_structure_id": best["structure_id"] if best else None,
            "best_predicted_smiles": best["predicted_smiles"] if best else None,
            "canonical_smiles_match": best["canonical_match"] if best else False,
            "inchikey_match": best["inchikey_match"] if best else False,
            "best_tanimoto_similarity": best["similarity"] if best else None,
            "tanimoto_at_or_above_threshold": (
                best["similarity"] >= TANIMOTO_THRESHOLD if best else False
            ),
        })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = OUTPUT_DIR / "ground_truth_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)

    n_predictions = len(matches_df)
    valid_predictions = int(matches_df["prediction_valid"].sum()) if n_predictions else 0
    valid_matches = matches_df[matches_df["prediction_valid"]]
    n_gt = len(gt_records)
    valid_gt_count = sum(g["gt_valid"] for g in gt_records)

    def rate(numerator, denominator):
        return round(float(numerator / denominator), 4) if denominator else None

    summary = {
        "configuration": {
            "predictions_csv": str(PREDICTIONS_CSV),
            "ground_truth_csv": str(GROUND_TRUTH_CSV),
            "tanimoto_threshold": TANIMOTO_THRESHOLD,
            "morgan_radius": MORGAN_RADIUS,
            "morgan_fingerprint_size": MORGAN_FP_SIZE,
            "paper_id_map": PAPER_ID_MAP,
        },
        "counts": {
            "predicted_structures": n_predictions,
            "valid_predictions": valid_predictions,
            "invalid_or_empty_predictions": n_predictions - valid_predictions,
            "ground_truth_molecules": n_gt,
            "valid_ground_truth_molecules": valid_gt_count,
        },
        "prediction_level_metrics": {
            "validity_rate": rate(valid_predictions, n_predictions),
            "exact_string_match_rate_all_predictions": rate(
                int(matches_df["exact_string_match"].sum()), n_predictions
            ),
            "canonical_match_rate_all_predictions": rate(
                int(matches_df["canonical_smiles_match"].sum()), n_predictions
            ),
            "inchikey_match_rate_all_predictions": rate(
                int(matches_df["inchikey_match"].sum()), n_predictions
            ),
            "tanimoto_threshold_match_rate_all_predictions": rate(
                int(matches_df["tanimoto_at_or_above_threshold"].sum()), n_predictions
            ),
            "canonical_match_rate_valid_predictions": rate(
                int(valid_matches["canonical_smiles_match"].sum()), len(valid_matches)
            ),
            "mean_best_tanimoto_valid_predictions": (
                round(float(valid_matches["best_tanimoto_similarity"].mean()), 4)
                if len(valid_matches) else None
            ),
        },
        "ground_truth_coverage_metrics": {
            "ground_truth_molecules_with_any_best_match": int(coverage_df["matched_by_prediction"].sum()),
            "coverage_rate_any_best_match": rate(
                int(coverage_df["matched_by_prediction"].sum()), n_gt
            ),
            "ground_truth_molecules_with_canonical_match": int(coverage_df["canonical_smiles_match"].sum()),
            "canonical_ground_truth_coverage_rate": rate(
                int(coverage_df["canonical_smiles_match"].sum()), n_gt
            ),
            "ground_truth_molecules_at_tanimoto_threshold": int(coverage_df["tanimoto_at_or_above_threshold"].sum()),
            "tanimoto_ground_truth_coverage_rate": rate(
                int(coverage_df["tanimoto_at_or_above_threshold"].sum()), n_gt
            ),
        },
        "per_paper": {},
        "notes": [
            "Each predicted structure is compared with its inferred paper's ground truth when possible; otherwise all ground-truth molecules are considered.",
            "Best-match comparisons are not one-to-one: multiple predictions may select the same ground-truth molecule.",
            "Tanimoto similarity is based on Morgan fingerprints and is not a probability of correctness.",
            "Exact string match is stricter than canonical SMILES or InChIKey match.",
        ],
    }

    for paper_id in sorted(coverage_df["paperID"].dropna().unique(), key=lambda x: str(x)):
        subset = coverage_df[coverage_df["paperID"] == paper_id]
        summary["per_paper"][str(paper_id)] = {
            "ground_truth_count": int(len(subset)),
            "canonical_matches": int(subset["canonical_smiles_match"].sum()),
            "canonical_coverage_rate": rate(int(subset["canonical_smiles_match"].sum()), len(subset)),
            "tanimoto_threshold_matches": int(subset["tanimoto_at_or_above_threshold"].sum()),
            "tanimoto_threshold_coverage_rate": rate(int(subset["tanimoto_at_or_above_threshold"].sum()), len(subset)),
        }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Evaluation complete.")
    print(f"Prediction-level results: {matches_path.resolve()}")
    print(f"Ground-truth coverage:    {coverage_path.resolve()}")
    print(f"Summary metrics:          {summary_path.resolve()}")
    print(json.dumps({
        "predicted_structures": n_predictions,
        "validity_rate": summary["prediction_level_metrics"]["validity_rate"],
        "canonical_match_rate_all_predictions": summary["prediction_level_metrics"]["canonical_match_rate_all_predictions"],
        "mean_best_tanimoto_valid_predictions": summary["prediction_level_metrics"]["mean_best_tanimoto_valid_predictions"],
        "canonical_ground_truth_coverage_rate": summary["ground_truth_coverage_metrics"]["canonical_ground_truth_coverage_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
