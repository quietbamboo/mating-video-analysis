# Chicken-label detector training

The images and YOLO labels are distributed in the public dataset rather than
this code repository. Extract them into `data/` using the layout described in
the top-level README, or pass another location with `--data-root`.

```bash
python -m pip install -r requirements.txt
python code/train.py --data-root data
```

The manuscript training settings are the defaults. Hardware-dependent values
can be overridden, for example:

```bash
python code/train.py --data-root /path/to/chicken-label \
  --device 0,1 --batch 24
```

