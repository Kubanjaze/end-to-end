"""Phase 100 — End-to-End: Paper to Extract to Evaluate to Review.

THE CAPSTONE. Complete pipeline: load compounds, compute RDKit descriptors,
train RF classifier, rank hits, Claude review of top-5.
"""
import sys
import os

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from anthropic import Anthropic
from dotenv import load_dotenv

RDLogger.DisableLog("rdApp.*")

env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

BANNER = r"""
================================================================================
    ____  _                        _    ___   ___
   |  _ \| |__   __ _ ___  ___   / |  / _ \ / _ \
   | |_) | '_ \ / _` / __|/ _ \  | | | | | | | | |
   |  __/| | | | (_| \__ \  __/  | | | |_| | |_| |
   |_|   |_| |_|\__,_|___/\___|  |_|  \___/ \___/

   END-TO-END PIPELINE: Data -> Descriptors -> ML -> Claude Review
================================================================================
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 100 — End-to-end: data -> descriptors -> ML -> Claude review",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="compounds.csv", help="Input CSV")
    p.add_argument("--top", type=int, default=5, help="Top compounds for Claude review")
    p.add_argument("--threshold", type=float, default=8.0, help="pIC50 hit threshold")
    p.add_argument("--model", default="claude-haiku-4-5-20251001", help="Claude model")
    p.add_argument("--output", default="output", help="Output directory")
    return p.parse_args()


def compute_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RDKit molecular descriptors and Morgan fingerprint."""
    records = []
    fps = []
    valid_idx = []

    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            print(f"    WARNING: invalid SMILES for {row['compound_name']}, skipping")
            continue

        desc = {
            "compound_name": row["compound_name"],
            "smiles": row["smiles"],
            "pic50": row["pic50"],
            "MolWt": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Descriptors.MolLogP(mol), 2),
            "HBA": Descriptors.NumHAcceptors(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "TPSA": round(Descriptors.TPSA(mol), 2),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "RingCount": Descriptors.RingCount(mol),
            "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        }
        records.append(desc)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps.append(np.array(fp))
        valid_idx.append(i)

    desc_df = pd.DataFrame(records)
    return desc_df, np.array(fps)


def train_and_rank(desc_df: pd.DataFrame, fps: np.ndarray, threshold: float) -> pd.DataFrame:
    """Train RF classifier, return ranked dataframe."""
    y = (desc_df["pic50"] >= threshold).astype(int).values
    n_hits = y.sum()
    print(f"    Compounds: {len(desc_df)}, Hits (pIC50 >= {threshold}): {n_hits}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    probs = cross_val_predict(clf, fps, y, cv=min(5, len(desc_df)), method="predict_proba")[:, 1]

    desc_df = desc_df.copy()
    desc_df["predicted_prob"] = probs
    desc_df["hit_label"] = y

    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    print(f"    Cross-val ROC-AUC: {auc:.3f}")

    return desc_df.sort_values("predicted_prob", ascending=False).reset_index(drop=True), auc


def claude_review(top_compounds: pd.DataFrame, model: str) -> list[dict]:
    """Send top compounds to Claude for final expert review."""
    client = Anthropic()

    compound_block = "\n".join(
        f"- {row['compound_name']}: SMILES={row['smiles']}, pIC50={row['pic50']:.2f}, "
        f"predicted_prob={row['predicted_prob']:.3f}, "
        f"MolWt={row['MolWt']}, LogP={row['LogP']}, HBA={row['HBA']}, "
        f"HBD={row['HBD']}, TPSA={row['TPSA']}, RotBonds={row['RotBonds']}"
        for _, row in top_compounds.iterrows()
    )

    prompt = f"""You are a senior medicinal chemist performing the final review of top hit candidates
from an end-to-end drug discovery pipeline (Phase 100 capstone).

These compounds were identified by:
1. Computing RDKit molecular descriptors
2. Training a Random Forest classifier on CETP inhibitor activity
3. Ranking by predicted hit probability

Top candidates:
{compound_block}

For each compound, provide a structured JSON array assessment. Each element:
- "compound_name": string
- "rank": int (1-based)
- "drug_likeness": "good" | "moderate" | "poor" (based on Lipinski/Veber rules)
- "risk_level": "low" | "medium" | "high"
- "recommendation": "advance" | "optimize" | "deprioritize"
- "rationale": string (2-3 sentences: structural assessment, property profile, development potential)

Return ONLY the JSON array."""

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    return json.loads(text)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(BANNER)

    # Stage 1: Load
    print("[Stage 1] Loading compound library...")
    df = pd.read_csv(args.data)
    print(f"    Loaded {len(df)} compounds from {args.data}")

    # Stage 2: Descriptors
    print("\n[Stage 2] Computing RDKit molecular descriptors...")
    desc_df, fps = compute_descriptors(df)
    desc_csv = out_dir / "descriptors.csv"
    desc_df.to_csv(desc_csv, index=False)
    print(f"    Computed descriptors for {len(desc_df)} compounds")
    print(f"    Saved to {desc_csv}")

    # Stage 3: ML classification
    print("\n[Stage 3] Training Random Forest classifier...")
    ranked_df, auc = train_and_rank(desc_df, fps, args.threshold)
    rank_csv = out_dir / "rankings.csv"
    ranked_df.to_csv(rank_csv, index=False)
    print(f"    Saved rankings to {rank_csv}")

    # Stage 4: Claude review
    top_n = ranked_df.head(args.top).copy()
    print(f"\n[Stage 4] Claude expert review of top-{args.top} compounds...")
    print(f"    Model: {args.model}")
    reviews = claude_review(top_n, args.model)
    print(f"    Received {len(reviews)} structured reviews")

    # Stage 5: Save report
    report = {
        "phase": 100,
        "description": "End-to-end pipeline: data -> descriptors -> ML -> Claude review",
        "pipeline_config": {
            "n_compounds": len(desc_df),
            "threshold": args.threshold,
            "n_hits": int((desc_df["pic50"] >= args.threshold).sum()),
            "roc_auc": round(auc, 3),
            "top_n_reviewed": args.top,
            "model": args.model,
        },
        "top_compounds": top_n[["compound_name", "smiles", "pic50", "predicted_prob"]].to_dict("records"),
        "claude_reviews": reviews,
    }
    report_path = out_dir / "end_to_end_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Final summary
    print(f"\n{'=' * 72}")
    print(f"  PHASE 100 COMPLETE — End-to-End Pipeline Summary")
    print(f"{'=' * 72}")
    print(f"  Compounds processed:    {len(desc_df)}")
    print(f"  Hit threshold:          pIC50 >= {args.threshold}")
    print(f"  Cross-val ROC-AUC:      {auc:.3f}")
    print(f"  Top-{args.top} reviewed by Claude ({args.model})")
    print()

    rec_counts = {}
    for r in reviews:
        rec = r.get("recommendation", "unknown")
        rec_counts[rec] = rec_counts.get(rec, 0) + 1
    print(f"  Recommendations:        {dict(rec_counts)}")

    print(f"\n  Outputs:")
    print(f"    {desc_csv}")
    print(f"    {rank_csv}")
    print(f"    {report_path}")
    print(f"{'=' * 72}")
    print("  Done.")


if __name__ == "__main__":
    main()
