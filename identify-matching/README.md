# Chicken identity matching

This component reproduces rooster and mating-hen identity attribution.

## Expected local layout

After downloading the public dataset and weights:

```text
identify-matching/
├── data/
│   ├── mating_clips.csv
│   ├── rooster_logs/
│   └── videos/
├── models/chicken-label-12-23.pt
├── src/chicken_identify/
├── scripts/
├── results/                 # generated locally
└── output/                  # generated locally
```

## Environment and data validation

Python 3.10 or newer is required. Run from `identify-matching/`:

```bash
python -m pip install -e .
python scripts/evaluate.py --validate-only
```

The validation command checks that every CSV record has a corresponding video
and rooster log without loading the YOLO model.

## Run the proposed method

The manuscript configuration uses a central ROI area ratio of `1/9`,
`tau_gap = 1.0 s`, an observation window of `Nw = 1.0F` valid points, and the
trajectory-distance filter `mu_d +/- 2 sigma_d`:

```bash
python scripts/evaluate.py \
  --roi-area-ratio 1/9 \
  --tau-gap-seconds 1.0 \
  --trajectory-window-fps-multiplier 1.0 \
  --outlier-sigma-multiplier 2 \
  --output results/generated/roi-results-1_9.csv
```

The fallback rule searches forward from the mating-interval midpoint and uses
the first valid reappearance when the main trajectory cannot determine the hen.

## Generate comparison results

```bash
python scripts/evaluate.py --method nearest-baseline \
  --output results/generated/nearest-baseline-results.csv

python scripts/evaluate.py --disable-tau-gap \
  --trajectory-window-fps-multiplier 1.0 \
  --output results/generated/ablation-no-tau-gap.csv

python scripts/evaluate.py --disable-trajectory-window \
  --output results/generated/ablation-no-observation-window.csv

python scripts/evaluate.py --disable-outlier-filter \
  --output results/generated/outlier-ablation-results-none.csv
```

Use `--detections-cache-dir output/detection-cache` to reuse label detections
when comparing settings.

## Analyze generated results

Results are intentionally recreated rather than stored in the code repository:

```bash
python scripts/analyze_results.py results/generated \
  --output-dir output/generated-analysis
```

The analysis script validates the event sets and writes accuracy, confidence
interval, paired McNemar, sensitivity, manuscript-value, LaTeX, and manifest
outputs.
