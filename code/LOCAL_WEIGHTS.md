# Local network weights (Hessian-affine stack)

The local branch uses three frozen networks from the upstream Norppa reference
implementation: `AffNet.pth`, `OriNet.pth` and `HardNet++.pth`.  They are **not**
redistributed in this archive because the upstream licence does not grant that.

Expected SHA-256 (from the frozen project record):

| file | sha256 |
|---|---|
| AffNet.pth | 65c8bb2cf71e913818522d09dcc4262bcbcef5d756baa86a41f28526c2d61160 |
| OriNet.pth | 030e7964ae3e136d00431c7e11c6b67df40ad299f02ac384fee0f1c3fd1dcb30 |
| HardNet++.pth | 9fe80a335fe72d1225fdd1ae0943418b12489f95014bd51dddc9819d6ffa33c7 |

Place them in a directory of your choice and point the scripts at it:

```
python code/stage_h_local_descriptors.py --subset holdout_gallery --weights-dir <dir>
```

The vendored code itself is in `third_party/local_fv/hesaff_pytorch/`.
