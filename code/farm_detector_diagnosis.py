# -*- coding: utf-8 -*-
"""Task 4 - why DET1 under-fires on farm video, and whether recognition or localization is limiting.

Diagnostics only: the production pipeline is untouched. The low-confidence sweep and the
counterfactual crops are ANALYSIS ARTEFACTS, not pipeline settings.
"""
import os, json, csv
from pathlib import Path
import numpy as np
import torch, cv2
from torchvision import transforms
from PIL import Image

os.environ.setdefault("TORCH_HOME", r"D:\H2025_clean_restart\models\global\_torch_home")
RUN = Path(r'D:\FarmReID_run')
OUT = Path(r'D:\CowReID_Paper\analysis'); OUT.mkdir(parents=True, exist_ok=True)
DET = Path(r'D:\H2025_clean_restart\analysis\stage_p2r_det1\runs\det1_yolov8n\weights\best.pt')

man = list(csv.DictReader(open(Path(r'D://FarmExternalEval//tmp_manifest.csv'), encoding='utf-8-sig')))
paths = [r['image_path'] for r in man]
ids = [r['identity'] for r in man]; splits = [r['split'] for r in man]
uids = sorted(set(ids)); ix = {v: i for i, v in enumerate(uids)}
gi = [i for i, r in enumerate(man) if r['split'] == 'gallery']
qi = [i for i, r in enumerate(man) if r['split'] == 'query']
true = np.array([ids[i] for i in qi])
print(f'frames={len(man)} identities={len(uids)} g={len(gi)} q={len(qi)}')

# ---------------- 1. detection failure attribution ----------------
from ultralytics import YOLO
model = YOLO(str(DET))

def sweep(conf):
    hits, maxc, nbox = 0, [], []
    for s in range(0, len(paths), 32):
        res = model.predict(source=paths[s:s+32], imgsz=640, conf=conf, iou=0.70, max_det=300, verbose=False)
        for r in res:
            if r.boxes is not None and len(r.boxes):
                c = r.boxes.conf.cpu().numpy()
                hits += 1; maxc.append(float(c.max())); nbox.append(int(len(c)))
            else:
                maxc.append(0.0); nbox.append(0)
    return hits, maxc, nbox

print('sweeping confidence thresholds (diagnostic only) ...')
sweepres = {}
for conf in (0.05, 0.10, 0.15, 0.20, 0.25):
    h, mc, nb = sweep(conf)
    sweepres[str(conf)] = {'frames_with_box': h, 'rate': round(h/len(paths), 4),
                           'max_conf_per_frame': mc}
    print(f'  conf {conf:.2f}: {h}/{len(paths)} = {h/len(paths)*100:.1f}%')

# geometry of the production boxes (conf 0.25)
roi = json.load(open(Path(r'D://FarmExternalEval//roi_44.json'), encoding='utf-8'))['records']
geo = []
for rec in roi:
    if not rec['detected']:
        continue
    im = cv2.imdecode(np.fromfile(rec['image_path'], np.uint8), cv2.IMREAD_COLOR)
    H, W = im.shape[:2]
    x0, y0, x1, y1 = rec['box']
    w, h = x1-x0, y1-y0
    geo.append({'image': Path(rec['image_path']).name, 'conf': rec['conf'],
                'area_frac': round(w*h/(W*H), 4), 'aspect': round(w/max(h,1), 3),
                'cy_frac': round((y0+y1)/2/H, 3), 'cx_frac': round((x0+x1)/2/W, 3),
                'box_px': [round(w), round(h)]})
ar = np.array([g['area_frac'] for g in geo]); cys = np.array([g['cy_frac'] for g in geo])
asp = np.array([g['aspect'] for g in geo]); cf = np.array([g['conf'] for g in geo])
geom = {'n': len(geo), 'area_frac_median': float(np.median(ar)), 'area_frac_min': float(ar.min()),
        'area_frac_max': float(ar.max()), 'small_lt_2pct': int((ar < 0.02).sum()),
        'aspect_median': float(np.median(asp)), 'portrait_box_lt_0.8': int((asp < 0.8).sum()),
        'centroid_y_median': float(np.median(cys)), 'centroid_y_gt_0.7': int((cys > 0.7).sum()),
        'conf_median': float(np.median(cf))}
print('box geometry:', json.dumps(geom, ensure_ascii=False))

# blur stratification
blur = []
for rec in roi:
    im = cv2.imdecode(np.fromfile(rec['image_path'], np.uint8), cv2.IMREAD_COLOR)
    h, w = im.shape[:2]; s = 320.0/w
    g = cv2.cvtColor(cv2.resize(im, (320, max(1, int(h*s)))), cv2.COLOR_BGR2GRAY)
    blur.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
blur = np.array(blur); det = np.array([bool(r['detected']) for r in roi])
q = np.quantile(blur, [0.25, 0.5, 0.75])
bands = {'Q1_blurriest': blur <= q[0], 'Q2': (blur > q[0]) & (blur <= q[1]),
         'Q3': (blur > q[1]) & (blur <= q[2]), 'Q4_sharpest': blur > q[2]}
blur_layer = {k: {'n': int(m.sum()), 'det_rate': round(float(det[m].mean()), 4)} for k, m in bands.items()}
print('detection rate by blur band:', json.dumps(blur_layer, ensure_ascii=False))

# ---------------- 2./3. recognition under localization conditions ----------------
IM_MEAN = [0.485, 0.456, 0.406]; IM_STD = [0.229, 0.224, 0.225]
def build_model(n_classes, emb=128):
    import torch.nn as nn
    from torchvision import models
    trunk = models.mobilenet_v2(weights=None).features
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.trunk = trunk; self.pool = nn.AdaptiveAvgPool2d(1)
            self.emb = nn.Linear(1280, emb); self.weight = nn.Parameter(torch.empty(n_classes, emb))
        def forward(self, x):
            import torch.nn.functional as F
            f = self.trunk(x); f = self.pool(f) if f.dim() > 2 else f
            return F.normalize(self.emb(torch.flatten(f, 1)), dim=1)
    return Net()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def prep(path, mode):
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if im is None: im = np.zeros((112,112,3), np.uint8)
    H, W = im.shape[:2]
    if mode == 'squash':
        return cv2.resize(im, (112,112))
    if mode == 'letterbox':
        s = 112/max(H, W); nw, nh = int(round(W*s)), int(round(H*s))
        r = cv2.resize(im, (nw, nh)); canvas = np.full((112,112,3), 114, np.uint8)
        oy, ox = (112-nh)//2, (112-nw)//2
        canvas[oy:oy+nh, ox:ox+nw] = r; return canvas
    if mode == 'topcenter60':
        ch, cw = int(H*0.6), int(W*0.6)
        y0 = int(H*0.15); x0 = (W-cw)//2
        return cv2.resize(im[y0:y0+ch, x0:x0+cw], (112,112))
    if mode == 'topcenter40':
        ch, cw = int(H*0.4), int(W*0.4)
        y0 = int(H*0.12); x0 = (W-cw)//2
        return cv2.resize(im[y0:y0+ch, x0:x0+cw], (112,112))
    raise ValueError(mode)

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD)])

def embed(paths_, mode, seed):
    ck = torch.load(rf'D:\H2025_clean_restart\models\global\mobilenetv2_s{seed}_img112_emb128\best.pt',
                    map_location='cpu', weights_only=False)
    net = build_model(ck['config']['n_classes']); net.load_state_dict(ck['state_dict']); net.eval().to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths_), 64):
            batch = torch.stack([tf(cv2.cvtColor(prep(p, mode), cv2.COLOR_BGR2RGB)) for p in paths_[i:i+64]]).to(dev)
            out.append(net(batch).cpu().numpy())
    return np.concatenate(out).astype('float32')

def cos(A, B):
    A = A/np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B/np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T

def ident_scores(S, glab):
    M = np.full((S.shape[0], len(uids)), -np.inf); gid = [ix[glab[b]] for b in range(len(glab))]
    for a in range(S.shape[0]):
        for b in range(S.shape[1]):
            if S[a, b] > M[a, gid[b]]: M[a, gid[b]] = S[a, b]
    return M

def r1_of(M, tvec):
    out = []
    for k, t in enumerate(tvec):
        s = M[k]; out.append(1.0 if s[ix[t]] >= s.max()-1e-12 else 0.0)
    return float(np.mean(out))

glab = np.array([ids[i] for i in gi])
loc = {}
for mode in ('squash', 'letterbox', 'topcenter60', 'topcenter40'):
    vals = []
    for seed in (42, 43, 44):
        E = embed(paths, mode, seed)
        S = cos(E[qi], E[gi]); vals.append(r1_of(ident_scores(S, glab), true))
    loc[mode] = {'rank1_mean3': round(float(np.mean(vals)), 4), 'per_seed': [round(v, 4) for v in vals]}
    print(f'  {mode:14} R1 {loc[mode]["rank1_mean3"]:.4f}  {loc[mode]["per_seed"]}')

# recognition by detector outcome (production pipeline features)
feat = np.load(Path(r'D:\FarmExternalEval\features_44.npz'))
det_q = det[qi]
sub = {'detected': det_q, 'fallback': ~det_q}
byloc = {}
for nm, m in sub.items():
    if m.sum() == 0: continue
    row = {'n_query': int(m.sum())}
    SG = cos(feat['E42'][qi][m], feat['E42'][gi])
    SL = cos(feat['fv'][qi][m], feat['fv'][gi])
    for tag, S in (('Global', SG), ('Local', SL), ('Fusion', 0.5*SG + 0.5*SL)):
        row[tag] = round(r1_of(ident_scores(S, glab), true[m]), 4)
    byloc[nm] = row
print('recognition by localization condition:', json.dumps(byloc, ensure_ascii=False))

res = {'protocol': {'frames': len(man), 'identities': len(uids), 'queries': len(qi),
                    'note': 'diagnostics only; production pipeline untouched'},
       'detection_confidence_sweep': {k: {kk: vv for kk, vv in v.items() if kk != 'max_conf_per_frame'}
                                      for k, v in sweepres.items()},
       'max_conf_distribution': {'median': float(np.median(sweepres['0.05']['max_conf_per_frame'])),
                                 'p90': float(np.percentile(sweepres['0.05']['max_conf_per_frame'], 90)),
                                 'zero_conf_frames': int((np.array(sweepres['0.05']['max_conf_per_frame']) == 0).sum())},
       'box_geometry_at_conf025': geom, 'detection_rate_by_blur_band': blur_layer,
       'recognition_by_localization': byloc, 'counterfactual_crops_global_rank1': loc}
(OUT/'farm_detector_diagnosis.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print('\nWROTE', OUT/'farm_detector_diagnosis.json')
