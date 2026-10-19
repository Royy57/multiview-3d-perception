# data/

Datasets are downloaded here and never committed.

```
python scripts/download_mipnerf360.py --scene counter --resolution 4
```

creates

```
data/mipnerf360/counter/
  images/             JPEGs at the chosen resolution
  reference_sparse/   the dataset's own COLMAP model (used only for evaluation)
  resolution.txt
```

Scenes: `bicycle`, `bonsai`, `counter`, `garden`, `kitchen`, `room`, `stump`
(the indoor ones, `bonsai`, `counter`, `kitchen` and `room`, have the most COCO objects).
