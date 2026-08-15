# Rooster detector training

The detection images and YOLO labels are distributed in the public dataset.
Extract the prepared split into `data/`, or pass another directory with
`--data-root`.

```bash
python -m pip install -r requirements.txt
python code/train.py --data-root data
```

For the original multi-GPU setting:

```bash
python code/train.py --data-root data --device 0,1 --batch 48
```
