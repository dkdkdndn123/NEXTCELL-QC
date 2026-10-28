"""
ParticleQC 도구: 프로젝트 폴더 관리 · 라벨링용 입자 자르기 · 모델 학습
프로젝트 폴더 구조
  photos/                 검사·학습에 쓴 사진 (복사본)
  labeling/_crops/        잘라낸 입자 (분류 전)
  labeling/sheets/        한눈에 보는 표 (번호 = crops.csv 순서)
  labeling/normal, abnormal, exclude/   사람이 옮겨 넣은 입자
  labeling/crops.csv      입자별 측정값 (file, image, sample, ecd_um, circularity, aspect_ratio, solidity)
  settings.json, model.npz, model_report.json, feat_cache.npz
"""
import os, glob, json, hashlib, shutil
import numpy as np, pandas as pd, cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import particle_engine as pe
from particle_model import features

LABEL_DIRS = {"normal": 0, "abnormal": 2, "exclude": 3}     # 모델 내부 분류 번호 (round=0, defect=2, exclude=3)
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def ensure_project(root):
    for d in ["photos", "labeling/_crops", "labeling/sheets", "labeling/normal", "labeling/abnormal", "labeling/exclude"]:
        os.makedirs(os.path.join(root, d), exist_ok=True)
    sp = os.path.join(root, "settings.json")
    if not os.path.exists(sp):
        pe.save_settings(sp, dict(pe.DEFAULTS))
    return sp


def project_status(root):
    lab = os.path.join(root, "labeling")
    cnt = {k: len([f for f in os.listdir(os.path.join(lab, k)) if f.endswith(".png")]) if os.path.isdir(os.path.join(lab, k)) else 0
           for k in ["_crops", "normal", "abnormal", "exclude"]}
    n_photos = len([f for f in os.listdir(os.path.join(root, "photos")) if f.lower().endswith(IMG_EXT)]) if os.path.isdir(os.path.join(root, "photos")) else 0
    has_model = os.path.exists(os.path.join(root, "model.npz"))
    return dict(photos=n_photos, crops_unsorted=cnt["_crops"], normal=cnt["normal"], abnormal=cnt["abnormal"], exclude=cnt["exclude"], model=has_model)


def ingest(root, photo_paths, s, sample="A", log=print):
    """사진들을 프로젝트에 복사하고 단독 입자를 잘라 _crops/ 와 crops.csv, sheets/ 를 만든다 (기존 것에 이어 붙임)"""
    lab = os.path.join(root, "labeling"); crops_dir = os.path.join(lab, "_crops")
    csv = os.path.join(lab, "crops.csv")
    old = pd.read_csv(csv, encoding="utf-8-sig") if os.path.exists(csv) else pd.DataFrame()
    rows = []
    for p in photo_paths:
        name = os.path.basename(p)
        dst = os.path.join(root, "photos", name)
        if os.path.abspath(p) != os.path.abspath(dst):
            shutil.copy(p, dst)
        if len(old) and (old["image"] == name).any():
            log(f"{name}: 이미 잘라낸 사진 — 건너뜀"); continue
        ov, df, crops = pe.analyze(dst, s, None, want_crops=True)
        n = 0
        stem = os.path.splitext(name)[0]
        for (_, r), c in zip(df.iterrows(), crops):
            if r["contact"]:
                continue
            fn = f"{sample}_{stem}_p{int(r['id']):04d}.png"
            ok, buf = cv2.imencode(".png", c); buf.tofile(os.path.join(crops_dir, fn))
            rows.append(dict(file=fn, image=name, sample=sample, cx=int(r["cx"]), cy=int(r["cy"]), ecd_um=r["ecd_um"],
                             circularity=r["circularity"], aspect_ratio=r["aspect_ratio"], solidity=r["solidity"]))
            n += 1
        log(f"{name}: 입자 {len(df)}개 중 단독 {n}개 잘라냄")
    new = pd.DataFrame(rows)
    allc = pd.concat([old, new], ignore_index=True) if len(old) else new
    allc.to_csv(csv, index=False, encoding="utf-8-sig")
    # 시트는 전체를 다시 만든다 (번호 = crops.csv 행 번호)
    for f in glob.glob(os.path.join(lab, "sheets", "sheet_*.png")):
        os.remove(f)
    C, R, T = 10, 6, 150
    for st in range(0, len(allc), C * R):
        sheet = np.full((R * (T + 4), C * (T + 4), 3), 255, np.uint8)
        for j, fn in enumerate(allc.file[st:st + C * R]):
            path = None
            for d in ["_crops", "normal", "abnormal", "exclude"]:
                q = os.path.join(lab, d, fn)
                if os.path.exists(q):
                    path = q; break
            if path is None:
                continue
            im = cv2.imdecode(np.fromfile(path, np.uint8), 0)
            im = cv2.cvtColor(cv2.resize(im, (T, T), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
            r, c = divmod(j, C); y, x = r * (T + 4), c * (T + 4); sheet[y:y + T, x:x + T] = im; t = str(st + j)
            cv2.rectangle(sheet, (x, y), (x + 8 + 10 * len(t), y + 17), (0, 0, 0), -1)
            cv2.putText(sheet, t, (x + 3, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(lab, "sheets", f"sheet_{st // (C * R):02d}.png"), sheet)
    log(f"합계 {len(allc)}개 (새로 {len(new)}개) · 시트 {(len(allc) + C * R - 1) // (C * R)}장")
    return allc


def _file_key(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def train(root, C=0.03, log=print):
    """labeling/normal, abnormal, exclude + crops.csv → model.npz, model_report.json. 사진 단위 5겹 교차검증 보고"""
    lab = os.path.join(root, "labeling")
    meta = pd.read_csv(os.path.join(lab, "crops.csv"), encoding="utf-8-sig")
    rows = []
    for cls, ci in LABEL_DIRS.items():
        d = os.path.join(lab, cls)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(".png"):
                rows.append((os.path.join(d, f), f, ci))
    if not rows:
        raise RuntimeError("normal / abnormal / exclude 폴더에 입자 그림이 없습니다. _crops 의 그림을 세 폴더로 옮겨 주세요.")
    d = pd.DataFrame(rows, columns=["path", "file", "y"]).merge(meta, on="file")
    if len(d) < len(rows):
        log(f"주의: crops.csv 에 없는 그림 {len(rows) - len(d)}개는 학습에서 제외")
    y = d.y.values; g = d.image.values
    counts = {k: int((y == v).sum()) for k, v in LABEL_DIRS.items()}
    log(f"라벨 수: {counts} · 사진 {len(set(g))}장")
    if counts["abnormal"] < 10 or counts["normal"] < 30:
        raise RuntimeError("abnormal 10개, normal 30개 이상은 있어야 학습이 됩니다. (권장: abnormal 100개, 전체 1,000개 이상)")
    shapes = d[["ecd_um", "circularity", "aspect_ratio", "solidity"]].values.astype(float)
    cache_p = os.path.join(root, "feat_cache.npz"); cache = {}
    if os.path.exists(cache_p):
        z = np.load(cache_p, allow_pickle=True); cache = dict(zip(z["keys"], z["feats"]))
    F = []
    nfeat = len(features(np.zeros((128, 128), np.uint8), shapes[0])[0])
    for i, (p, s4) in enumerate(zip(d.path, shapes)):
        k = _file_key(p)
        if k in cache and cache[k].shape[1] == nfeat:
            F.append(cache[k])
        else:
            F.append(features(cv2.imdecode(np.fromfile(p, np.uint8), 0), s4)); cache[k] = F[-1]
        if (i + 1) % 200 == 0:
            log(f"특징 추출 {i + 1}/{len(d)}")
    np.savez(cache_p, keys=np.array(list(cache.keys())), feats=np.array(list(cache.values()), dtype=object))

    def fit(idx):
        X = np.vstack([F[i] for i in idx]); Y = np.repeat(y[idx], 8)
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=C, max_iter=8000, class_weight="balanced").fit(sc.transform(X), Y)
        return sc, m

    def proba4(m, X):
        P = np.zeros((len(X), 4)); P[:, list(m.classes_)] = m.predict_proba(X); return P

    n_img = len(set(g)); folds = min(5, n_img)
    P = np.zeros((len(y), 4))
    if folds >= 2:
        for k, (tr, te) in enumerate(GroupKFold(folds).split(F, y, g)):
            sc, m = fit(tr)
            for i in te:
                P[i] = proba4(m, sc.transform(F[i])).mean(0)
            log(f"교차검증 {k + 1}/{folds}")
    else:
        sc, m = fit(np.arange(len(y)))
        for i in range(len(y)):
            P[i] = proba4(m, sc.transform(F[i])).mean(0)
        log("사진이 1장뿐이라 교차검증 대신 학습 데이터로 평가 (참고용)")
    yd = (y == 2).astype(int)
    ths = np.arange(0.2, 0.81, 0.02)
    cands = []
    for tx in ([0.3, 0.4, 0.5] if counts["exclude"] > 0 else [1.01]):
        keep = P[:, 3] < tx
        for td in ths:
            pd_ = (keep & (P[:, 2] >= td)).astype(int)
            cands.append((f1_score(yd, pd_), abs(int(pd_.sum()) - int(yd.sum())), tx, float(td)))
    f1max = max(c[0] for c in cands)
    _, _, thr_ex, thr_def = min((c for c in cands if c[0] >= f1max - 0.01), key=lambda c: (c[1], -c[0]))
    pred = np.zeros(len(P), int); pred[P[:, 2] >= thr_def] = 2; pred[P[:, 3] >= thr_ex] = 3
    pd_ = (pred == 2).astype(int); lab_ok = y < 3
    rep = dict(n=int(len(y)), per_class=counts, n_images=int(n_img), thr_exclude=float(thr_ex), thr_defect=float(thr_def), thr_nonspherical=1.01,
               confusion=confusion_matrix(y, pred, labels=[0, 2, 3]).tolist(),
               abnormal=dict(precision=float(precision_score(yd, pd_, zero_division=0)), recall=float(recall_score(yd, pd_, zero_division=0)),
                             f1=float(f1_score(yd, pd_, zero_division=0)), n_true=int(yd.sum()), n_pred=int(pd_.sum())),
               exclude=dict(caught=float((pred[y == 3] == 3).mean()) if counts["exclude"] else None, labeled_lost=float((pred[lab_ok] == 3).mean())))
    if "sample" in d.columns:
        per = {}
        for smp in sorted(d["sample"].astype(str).unique()):
            m_ = (d["sample"].astype(str) == smp).values; judged = m_ & (pred < 3); labeled = m_ & lab_ok
            per[smp] = dict(n=int(m_.sum()), abnormal_model=float(100 * (pred[judged] == 2).mean()) if judged.any() else None,
                            abnormal_label=float(100 * (y[labeled] == 2).mean()) if labeled.any() else None)
        rep["per_sample"] = per
    sc, m = fit(np.arange(len(y)))
    W = np.zeros((4, m.coef_.shape[1])); b = np.full(4, -1e6)
    if len(m.classes_) == 2:
        W[m.classes_[1]] = m.coef_[0]; b[m.classes_[1]] = m.intercept_[0]; b[m.classes_[0]] = 0
    else:
        W[list(m.classes_)] = m.coef_; b[list(m.classes_)] = m.intercept_
    np.savez(os.path.join(root, "model.npz"), mean=sc.mean_, scale=sc.scale_, coef=W, bias=b,
             thr_exclude=thr_ex, thr_defect=thr_def, thr_nonspherical=1.01)
    json.dump(rep, open(os.path.join(root, "model_report.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log("모델 저장: model.npz")
    return rep
