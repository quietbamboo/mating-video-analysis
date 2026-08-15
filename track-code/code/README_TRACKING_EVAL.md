# BoT-SORT trajectory evaluation

The evaluator uses TrackEval 1.3.0 to calculate HOTA, CLEAR MOT, and Identity
metrics for sequences shared by:

```text
mot_dataset/<sequence>/gt/gt.txt       # generated prediction
mot_dataset_true/<sequence>/gt/gt.txt  # manual ground truth from public data
```

Although the prediction is named `gt.txt` for CVAT compatibility, the script
treats it as tracker output.

Run from `track-code/` after generating `mot_dataset/`:

```bash
python -m pip install -r code/requirements.txt
python code/evaluate_mot_tracking.py
```

Or without changing the current environment:

```bash
uv run --with trackeval==1.3.0 python code/evaluate_mot_tracking.py
```

The default IoU threshold for CLEAR and Identity metrics is 0.5. HOTA is
averaged over its standard 19 thresholds from 0.05 to 0.95. Outputs are:

```text
mot_evaluation/tracking_metrics.csv
mot_evaluation/tracking_metrics.json
```

Custom paths can be supplied with `--predictions`, `--ground-truth`, and
`--output-dir`. Prediction track IDs do not need to equal ground-truth IDs;
TrackEval performs identity matching within each sequence.

Only independently checked manual annotations should be used as ground truth.
If model trajectories were used to initialize annotation, inspect every frame
for missing boxes, false positives, and identity switches before evaluation.

