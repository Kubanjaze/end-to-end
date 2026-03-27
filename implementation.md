# Phase 100 — End-to-End: Paper to Extract to Evaluate to Review

## Version: 1.0 (Plan)

## Goal
The capstone phase. Build a complete end-to-end pipeline that: loads compounds, computes RDKit descriptors, trains an RF classifier, ranks hits, and sends the top-5 to Claude for structured review. Combines data loading, cheminformatics, ML, and LLM review into a single cohesive pipeline.

## CLI
```bash
PYTHONUTF8=1 python main.py
PYTHONUTF8=1 python main.py --top 5 --threshold 8.0 --model claude-haiku-4-5-20251001
```

## Outputs
- `output/descriptors.csv` — computed RDKit descriptors for all compounds
- `output/rankings.csv` — compounds ranked by predicted hit probability
- `output/end_to_end_report.json` — final report with Claude review of top-5
- Console: Phase 100 banner + pipeline stages + summary

## Logic
1. Print Phase 100 banner
2. Load `compounds.csv` (compound_name, smiles, pic50)
3. Compute RDKit molecular descriptors (MolWt, LogP, HBA, HBD, TPSA, RotBonds, RingCount, Morgan FP)
4. Binarize activity: pic50 >= threshold -> hit
5. Train Random Forest classifier (cross-validated predictions)
6. Rank by predicted hit probability
7. Select top-N compounds
8. Send top-N to Claude for structured review (one API call)
9. Save all outputs
10. Print summary

## Key Concepts
- Full pipeline integration: data -> features -> ML -> LLM review
- RDKit descriptor computation (not just fingerprints)
- RF classification with cross-validation
- Claude structured output for final expert review
- Single lean API call

## Verification Checklist
- [ ] `--help` works
- [ ] Phase 100 banner prints
- [ ] Descriptors computed for all valid compounds
- [ ] Classifier trains and produces rankings
- [ ] Claude API call succeeds
- [ ] All output files well-formed
- [ ] Console shows full pipeline summary

## Risks
- Small dataset (45 compounds) — acceptable for demonstration
- Claude review quality depends on context provided — mitigated by including descriptors + predictions
