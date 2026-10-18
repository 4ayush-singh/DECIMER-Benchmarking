from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import traceback

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm

from DECIMER import predict_SMILES
from decimer_segmentation import segment_chemical_structures
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.warning")

# ---------- CONFIGURATION ----------
INPUT_DIR = Path("pdfs")
OUTPUT_DIR = Path("output")
STRUCTURE_DIR = OUTPUT_DIR / "structures"
PREVIEW_DIR = OUTPUT_DIR / "page_previews"

DPI = 300
SAVE_PAGE_PREVIEWS = True

RESULTS_CSV = OUTPUT_DIR / "results.csv"
RESULTS_JSONL = OUTPUT_DIR / "results.jsonl"
FAILURES_CSV = OUTPUT_DIR / "failures.csv"
MANIFEST_JSON = OUTPUT_DIR / "manifest.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def smiles_properties(smiles):
    """Validate a predicted SMILES and derive normalized identifiers."""
    if not smiles or not str(smiles).strip():
        return {
            "valid_smiles": False,
            "canonical_smiles": None,
            "inchikey": None,
            "formula": None,
            "molecular_weight": None,
            "error": "Empty SMILES",
        }

    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return {
            "valid_smiles": False,
            "canonical_smiles": None,
            "inchikey": None,
            "formula": None,
            "molecular_weight": None,
            "error": "RDKit could not parse SMILES",
        }

    return {
        "valid_smiles": True,
        "canonical_smiles": Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        ),
        "inchikey": Chem.MolToInchiKey(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": rdMolDescriptors.CalcExactMolWt(mol),
        "error": None,
    }


def save_incrementally(record, failure=None):
    """Append each completed result immediately."""
    pd.DataFrame([record]).to_csv(
        RESULTS_CSV,
        mode="a",
        header=not RESULTS_CSV.exists(),
        index=False,
    )

    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if failure is not None:
        pd.DataFrame([failure]).to_csv(
            FAILURES_CSV,
            mode="a",
            header=not FAILURES_CSV.exists(),
            index=False,
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(INPUT_DIR.rglob("*.pdf"))
    pdfs += sorted(INPUT_DIR.rglob("*.PDF"))
    pdfs = sorted(set(pdfs))

    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {INPUT_DIR.resolve()}"
        )

    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(INPUT_DIR.resolve()),
        "pdf_count": len(pdfs),
        "dpi": DPI,
        "files": [],
    }

    for pdf in tqdm(pdfs, desc="PDFs"):
        relative_pdf = pdf.relative_to(INPUT_DIR)
        paper_id = hashlib.sha256(
            str(relative_pdf).encode("utf-8")
        ).hexdigest()[:12]

        paper_output = STRUCTURE_DIR / paper_id
        paper_output.mkdir(parents=True, exist_ok=True)

        pdf_info = {
            "paper_id": paper_id,
            "relative_pdf": str(relative_pdf),
            "pdf_sha256": sha256_file(pdf),
            "status": "processing",
            "structures_detected": 0,
            "structures_processed": 0,
            "errors": [],
        }

        try:
            # Use a temporary list per document to avoid rendering
            # the entire PDF into memory at once.
            from pdf2image import pdfinfo_from_path
            page_count = int(pdfinfo_from_path(str(pdf))["Pages"])

            for page_number in range(1, page_count + 1):
                try:
                    pages = convert_from_path(
                        str(pdf),
                        dpi=DPI,
                        first_page=page_number,
                        last_page=page_number,
                    )
                    if not pages:
                        raise RuntimeError("No page image returned")

                    page_pil = pages[0].convert("RGB")

                    if SAVE_PAGE_PREVIEWS:
                        preview_path = (
                            PREVIEW_DIR /
                            f"{paper_id}_page_{page_number:04d}.jpg"
                        )
                        page_pil.save(preview_path, quality=90)

                    # The segmentation example uses OpenCV images.
                    page_array = np.array(page_pil)
                    page_bgr = cv2.cvtColor(
                        page_array, cv2.COLOR_RGB2BGR
                    )

                    detection_start = time.perf_counter()

                    segments = segment_chemical_structures(
                        page_bgr,
                        expand=True,
                    )

                    detection_seconds = (
                        time.perf_counter() - detection_start
                    )

                    pdf_info["structures_detected"] += len(segments)

                    for structure_number, segment in enumerate(segments, 1):
                        structure_id = (
                            f"{paper_id}_p{page_number:04d}_"
                            f"s{structure_number:04d}"
                        )
                        image_path = paper_output / f"{structure_id}.png"

                        row = {
                            "structure_id": structure_id,
                            "paper_id": paper_id,
                            "source_pdf": str(relative_pdf),
                            "pdf_sha256": pdf_info["pdf_sha256"],
                            "page_number": page_number,
                            "structure_number_on_page": structure_number,
                            "image_path": str(image_path),
                            "dpi": DPI,
                            "detection_seconds_page": detection_seconds,
                            "recognition_seconds": None,
                            "predicted_smiles": None,
                            "valid_smiles": False,
                            "canonical_smiles": None,
                            "inchikey": None,
                            "formula": None,
                            "molecular_weight": None,
                            "error": None,
                            "processed_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        }

                        try:
                            if segment is None or segment.size == 0:
                                raise ValueError("Empty structure crop")

                            # Segmentation examples return image arrays.
                            if segment.ndim == 2:
                                crop = Image.fromarray(segment)
                            else:
                                # OpenCV convention: BGR -> RGB.
                                crop = Image.fromarray(
                                    cv2.cvtColor(
                                        segment, cv2.COLOR_BGR2RGB
                                    )
                                )

                            crop.save(image_path)

                            start = time.perf_counter()
                            smiles = predict_SMILES(str(image_path))
                            row["recognition_seconds"] = (
                                time.perf_counter() - start
                            )
                            row["predicted_smiles"] = (
                                str(smiles).strip()
                                if smiles is not None else None
                            )

                            row.update(
                                smiles_properties(row["predicted_smiles"])
                            )

                        except Exception as exc:
                            row["error"] = (
                                f"{type(exc).__name__}: {exc}"
                            )
                            pdf_info["errors"].append({
                                "structure_id": structure_id,
                                "error": row["error"],
                            })

                            failure = {
                                "paper_id": paper_id,
                                "source_pdf": str(relative_pdf),
                                "page_number": page_number,
                                "structure_id": structure_id,
                                "image_path": str(image_path),
                                "error": row["error"],
                                "traceback": traceback.format_exc(),
                            }
                            save_incrementally(row, failure)
                        else:
                            save_incrementally(row)

                        pdf_info["structures_processed"] += 1

                except Exception as exc:
                    pdf_info["errors"].append({
                        "page_number": page_number,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

            pdf_info["status"] = "completed_with_errors" if (
                pdf_info["errors"]
            ) else "completed"

        except Exception as exc:
            pdf_info["status"] = "failed"
            pdf_info["errors"].append({
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })

        manifest["files"].append(pdf_info)

        # Preserve document-level progress after every PDF.
        manifest["updated_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        MANIFEST_JSON.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("\nExtraction complete.")
    print(f"Results: {RESULTS_CSV.resolve()}")
    print(f"JSONL:   {RESULTS_JSONL.resolve()}")
    print(f"Failures: {FAILURES_CSV.resolve()}")
    print(f"Manifest: {MANIFEST_JSON.resolve()}")


if __name__ == "__main__":
    main()