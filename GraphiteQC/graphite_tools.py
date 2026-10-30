"""
GraphiteQC 다시 학습 도구 (NEXT CELL)
- 새 ×2,000 사진에서 단독 입자를 잘라 내고(라벨링 준비), 사람이 분류한 입자로 AI 판정 모델을 다시 학습한다.
- 검사 화면의 'AI 다시 학습'과 명령(ingest_new_photos.py · train_particle_model.py)이 같은 함수를 쓴다.
학습 폴더 구조 (명령으로 만든 폴더와 같음)
  crops/                                    잘라 낸 입자 (분류 전)
  round/ nonspherical/ defect/ exclude/     사람이 옮겨 넣은 입자
  crops.csv                                 입자별 측정값 (file, image, sample, cx, cy, ecd_um, circularity, aspect_ratio, solidity)
  sheet_XX.png                              한눈에 보는 표 (번호 = crops.csv 순서)
  photos/<시료 이름>/                        화면에서 올린 사진 (복사본)
  particle_model.npz · particle_model_report.json   화면에서 학습한 새 모델 (검사에 쓰려면 '새 모델로 바꾸기')
"""
import os, sys, glob, math, json, hashlib, datetime
import numpy as np, pandas as pd, cv2
from skimage.measure import regionprops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graphite_engine as ge
from particle_model import features, CLASSES

SHEET_COLS, SHEET_ROWS, SHEET_TILE = 10, 6, 150
MIN_DEFECT, MIN_NORMAL, MIN_PHOTOS = 10, 30, 2          # 이보다 적으면 학습하지 않음
REC_DEFECT, REC_TOTAL = 100, 1000                       # 권장 (사용법.txt)


# ---------------------------------------------------------------- 학습 폴더
def ensure_label_dir(lab):
    for d in ["crops"] + CLASSES:
        os.makedirs(os.path.join(lab, d), exist_ok=True)
    return lab


def _pngs(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".png")) if os.path.isdir(d) else []


def _read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def label_status(lab):
    cnt = {k: len(_pngs(os.path.join(lab, k))) for k in ["crops"] + CLASSES}
    csv = os.path.join(lab, "crops.csv")
    photos = int(_read_csv(csv)["image"].nunique()) if os.path.exists(csv) else 0
    return dict(photos=photos, unsorted=cnt["crops"], model=os.path.exists(os.path.join(lab, "particle_model.npz")),
                **{k: cnt[k] for k in CLASSES})


# ---------------------------------------------------------------- 입자 잘라 내기
def crop_singles(path):
    """사진 한 장 → [(128x128 입자 그림, 측정값 dict), ...]
    붙은·응집 입자, 테이프 주름, 사진 가장자리에 걸린 입자는 뺀다 (v1.0.3 ingest_new_photos.py 와 같은 규칙)"""
    raw = ge.read_gray(path)
    if raw is None:
        raise ValueError(f"사진을 읽지 못했습니다: {os.path.basename(path)}")
    ppu = ge.px_per_um(2000, raw.shape[1]); img = ge.split_info_bar(raw)
    scale = ge.REF_WIDTH / raw.shape[1]
    if abs(scale - 1) > 0.01:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC); ppu = ge.MAG_SETTINGS[2000][1]
    bg = ge.background(img, ppu); labels = ge.segment(img, ppu, bg)
    flat = img.astype(np.float32) - bg.astype(np.float32); cfrac = ge.contact_fraction(labels)
    ring_k = max(5, int(0.6 * ppu) | 1); h, w = img.shape
    out = []
    for rp in regionprops(labels):
        minr, minc, maxr, maxc = rp.bbox
        if minr == 0 or minc == 0 or maxr >= h or maxc >= w: continue
        ecd = math.sqrt(4 * rp.area / math.pi) / ppu
        if ecd < ge.MIN_ECD_UM or ecd > ge.MAX_ECD_UM: continue
        mask = (labels[minr:maxr, minc:maxc] == rp.label).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cnt = max(cnts, key=cv2.contourArea)
        perim = cv2.arcLength(cv2.approxPolyDP(cnt, 1.0, True), True)
        circ = min(1.0, 4 * math.pi * rp.area / perim ** 2) if perim > 0 else 0
        ar = rp.axis_major_length / rp.axis_minor_length if rp.axis_minor_length > 0 else 99
        pad = ring_k; y0, x0 = max(0, minr - pad), max(0, minc - pad)
        sub = (labels[y0:maxr + pad, x0:maxc + pad] == rp.label).astype(np.uint8)
        ring = cv2.dilate(sub, np.ones((ring_k, ring_k), np.uint8)) - sub; fsub = flat[y0:maxr + pad, x0:maxc + pad]
        contrast = float(fsub[sub > 0].mean() - fsub[ring > 0].mean()) if ring.any() else 99.0
        if ge.is_not_particle(contrast, circ, rp.solidity): continue
        if cfrac.get(rp.label, 0) > ge.AGG_CONTACT or rp.solidity < ge.AGG_SOL or ecd > 25: continue
        out.append((ge.crop_particle(img, rp),
                    dict(label=int(rp.label), cx=int(rp.centroid[1]), cy=int(rp.centroid[0]), ecd_um=round(ecd, 2),
                         circularity=round(circ, 3), aspect_ratio=round(ar, 3), solidity=round(rp.solidity, 3))))
    return out


def _find_crop(lab, name):
    for d in ["crops"] + CLASSES:
        q = os.path.join(lab, d, name)
        if os.path.exists(q):
            return q
    return None


def write_sheets(lab, allc):
    """번호가 붙은 시트(sheet_XX.png)를 전체 다시 만든다. 번호 = crops.csv 행 번호. 이미 분류한 입자도 찾아서 넣는다"""
    for f in glob.glob(os.path.join(lab, "sheet_*.png")):
        os.remove(f)
    C, R, T = SHEET_COLS, SHEET_ROWS, SHEET_TILE
    paths = []
    for s in range(0, len(allc), C * R):
        sheet = np.full((R * (T + 4), C * (T + 4), 3), 255, np.uint8)
        for j, name in enumerate(allc.file[s:s + C * R]):
            p = _find_crop(lab, name)
            if p is None:
                continue
            im = cv2.imdecode(np.fromfile(p, np.uint8), 0)
            im = cv2.cvtColor(cv2.resize(im, (T, T), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
            r, c = divmod(j, C); y, x = r * (T + 4), c * (T + 4); sheet[y:y + T, x:x + T] = im; t = str(s + j)
            cv2.rectangle(sheet, (x, y), (x + 8 + 10 * len(t), y + 17), (0, 0, 0), -1)
            cv2.putText(sheet, t, (x + 3, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        p = os.path.join(lab, f"sheet_{s // (C * R):02d}.png")
        ok, buf = cv2.imencode(".png", sheet); buf.tofile(p)
        paths.append(p)
    return paths


def ingest(lab, photo_paths, sample="new", log=print):
    """사진들의 단독 입자를 잘라 crops/ 에 넣고 crops.csv · 시트를 만든다.
    이미 있는 학습 폴더면 이어 붙인다 (같은 시료 이름의 같은 사진은 건너뜀)"""
    ensure_label_dir(lab)
    csv = os.path.join(lab, "crops.csv")
    old = _read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    if len(old) and "sample" not in old.columns:
        old.insert(2, "sample", "")
    done = set(zip(old["sample"].fillna("").astype(str), old["image"].astype(str))) if len(old) else set()
    rows = []
    for p in photo_paths:
        name = os.path.basename(p)
        if (sample, name) in done:
            log(f"{name}: 이미 잘라 낸 사진이라 건너뜀")
            continue
        stem = os.path.splitext(name)[0]
        n = 0
        for crop, r in crop_singles(p):
            fn = f"{sample}_{stem}_p{r['label']:04d}.png"
            ok, buf = cv2.imencode(".png", crop); buf.tofile(os.path.join(lab, "crops", fn))
            rows.append(dict(file=fn, image=name, sample=sample, cx=r["cx"], cy=r["cy"], ecd_um=r["ecd_um"],
                             circularity=r["circularity"], aspect_ratio=r["aspect_ratio"], solidity=r["solidity"]))
            n += 1
        done.add((sample, name))
        log(f"{name}: 단독 입자 {n}개")
    new = pd.DataFrame(rows)
    allc = pd.concat([old, new], ignore_index=True) if len(old) else new
    allc.to_csv(csv, index=False, encoding="utf-8-sig")
    sheets = write_sheets(lab, allc) if len(allc) else []
    log(f"합계 {len(allc)}개 (새로 {len(new)}개) · 시트 {len(sheets)}장")
    return allc, sheets


# ---------------------------------------------------------------- 학습
def _file_key(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _decide(P, thr_ns, thr_def, thr_ex):
    out = np.zeros(len(P), int)
    out[P[:, 1] >= thr_ns] = 1
    out[P[:, 2] >= thr_def] = 2
    out[P[:, 3] >= thr_ex] = 3
    return out


def _metrics(y, pred, samples):
    """같은 라벨에서 결함 찾기 성능과 시료별 결함 비율 (지금 모델과 새 모델을 같은 잣대로 비교)"""
    yd = (y == 2).astype(int); pdd = (pred == 2).astype(int); lab = y < 3
    tp = int((yd & pdd).sum())
    m = dict(recall=tp / max(int(yd.sum()), 1), precision=tp / max(int(pdd.sum()), 1) if pdd.sum() else 0.0,
             n_true=int(yd.sum()), n_pred=int(pdd.sum()), labeled_lost=float((pred[lab] == 3).mean()) if lab.any() else 0.0)
    if samples is not None:
        per = {}
        for s in sorted(pd.unique(samples)):
            ms = samples == s; judged = ms & (pred < 3); labeled = ms & lab
            per[str(s)] = dict(model=float(100 * (pred[judged] == 2).mean()) if judged.any() else None,
                               label=float(100 * (y[labeled] == 2).mean()) if labeled.any() else None)
        m["per_sample"] = per
    return m


def _proba_with(npz_path, F):
    """저장된 모델로 같은 특징의 확률을 계산 (ParticleModel.predict 와 같은 계산)"""
    with np.load(npz_path) as z:
        mean, scale, coef, bias = z["mean"], z["scale"], z["coef"], z["bias"]
        thr = (float(z["thr_nonspherical"]), float(z["thr_defect"]), float(z["thr_exclude"]))
    if mean.shape[0] != F[0].shape[1]:
        raise ValueError("특징 개수가 달라 비교할 수 없습니다")
    P = np.zeros((len(F), 4))
    for i, f in enumerate(F):
        Z = ((f - mean) / scale) @ coef.T + bias
        Z -= Z.max(1, keepdims=True); E = np.exp(Z); E /= E.sum(1, keepdims=True)
        P[i] = E.mean(0)
    return P, thr


def train(lab, out, cache=None, compare=None, log=print, progress=None, C=0.03):
    """lab 의 round / nonspherical / defect / exclude + crops.csv 로 4분류 모델을 학습해 out(.npz)과 _report.json 을 만든다.
    - 사진 단위 교차검증(최대 5겹)으로 성능을 낸 뒤, 전체 데이터로 최종 모델 학습 (v1.0.3 train_particle_model.py 와 같은 계산)
    - compare: 지금 쓰는 모델(.npz). 주면 같은 라벨에서 그 모델의 성능도 계산해 보고서에 넣는다"""
    tick = progress or (lambda frac, desc: None)
    meta = _read_csv(os.path.join(lab, "crops.csv"))
    rows = []
    for ci, cls in enumerate(CLASSES):
        for f in _pngs(os.path.join(lab, cls)):          # 비어 있거나 없는 분류(예: 비구형을 쓰지 않을 때)도 허용
            rows.append((os.path.join(lab, cls, f), f, ci))
    d = pd.DataFrame(rows, columns=["path", "file", "y"]).merge(meta, on="file")
    if len(d) < len(rows):
        log(f"주의: crops.csv 에 없는 그림 {len(rows) - len(d)}개는 학습에서 뺐습니다 (다른 폴더에서 옮겨 온 그림이면 그 폴더의 crops.csv 줄도 함께 있어야 합니다)")
    y = d.y.values; g = d.image.values
    counts = dict(zip(CLASSES, np.bincount(y, minlength=4).tolist()))
    n_img = len(set(g))
    log(f"라벨: round {counts['round']} · nonspherical {counts['nonspherical']} · defect {counts['defect']} · exclude {counts['exclude']} · 사진 {n_img}장")
    if counts["defect"] < MIN_DEFECT or counts["round"] + counts["nonspherical"] < MIN_NORMAL or n_img < MIN_PHOTOS:
        raise RuntimeError(f"학습하려면 defect {MIN_DEFECT}개, 정상(round + nonspherical) {MIN_NORMAL}개, 사진 {MIN_PHOTOS}장 이상이 필요합니다 "
                           f"(권장: defect {REC_DEFECT}개 이상, 전체 {REC_TOTAL:,}개 이상)")
    shapes = d[["ecd_um", "circularity", "aspect_ratio", "solidity"]].values.astype(float)

    # 특징 추출 (캐시가 있으면 파일 내용 해시로 재사용)
    cache_d = {}
    if cache and os.path.exists(cache):
        with np.load(cache, allow_pickle=True) as z:          # 바로 닫아야 아래에서 같은 파일에 다시 쓸 수 있음 (Windows)
            cache_d = dict(zip(z["keys"], z["feats"]))
    nfeat = len(features(np.zeros((128, 128), np.uint8), shapes[0])[0])
    F = []
    for i, (p, s) in enumerate(zip(d.path, shapes)):
        k = _file_key(p)
        if k in cache_d and cache_d[k].shape[1] == nfeat:
            F.append(cache_d[k])
        else:
            F.append(features(cv2.imdecode(np.fromfile(p, np.uint8), 0), s)); cache_d[k] = F[-1]
        if (i + 1) % 200 == 0 or i + 1 == len(d):
            tick(0.4 * (i + 1) / len(d), f"특징 추출 {i + 1}/{len(d)}")
            if (i + 1) % 500 == 0:
                log(f"특징 추출 {i + 1}/{len(d)}")
    if cache:
        np.savez(cache, keys=np.array(list(cache_d.keys())), feats=np.array(list(cache_d.values()), dtype=object))

    def fit(idx):
        X = np.vstack([F[i] for i in idx]); Y = np.repeat(y[idx], 8)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=C, max_iter=8000, class_weight="balanced").fit(sc.transform(X), Y)
        return sc, m

    def proba4(m, X):
        """라벨에 없는 분류가 있어도 항상 4열(round/nonspherical/defect/exclude) 확률로 맞춘다"""
        P = np.zeros((len(X), 4)); P[:, list(m.classes_)] = m.predict_proba(X)
        return P

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef
    folds = min(5, n_img)
    P = np.zeros((len(y), 4))
    for k, (tr, te) in enumerate(GroupKFold(folds).split(F, y, g)):
        sc, m = fit(tr)
        for i in te:
            P[i] = proba4(m, sc.transform(F[i])).mean(0)
        log(f"교차검증 {k + 1}/{folds}")
        tick(0.4 + 0.45 * (k + 1) / folds, f"교차검증 {k + 1}/{folds}")

    yd = (y == 2).astype(int)
    ths = np.arange(0.2, 0.81, 0.02)
    cands = []
    for tx in [0.3, 0.4, 0.5]:
        keep = P[:, 3] < tx
        for td in ths:
            pd_ = (keep & (P[:, 2] >= td)).astype(int)
            cands.append((f1_score(yd, pd_), abs(int(pd_.sum()) - int(yd.sum())), tx, float(td)))
    if (y == 3).sum() == 0:                       # 제외 라벨이 없으면 판정불가를 끈다
        cands = [c for c in cands if c[2] == 0.5]
    f1max = max(c[0] for c in cands)
    # 결함 F1이 최대와 0.01 이내인 기준값 중, 예측 결함 수가 실제 결함 수와 가장 가까운 것(시료 결함 비율이 치우치지 않게)
    _, _, thr_ex, thr_def = min((c for c in cands if c[0] >= f1max - 0.01), key=lambda c: (c[1], -c[0]))
    keep = P[:, 3] < thr_ex
    nd = keep & (y < 2)
    yn = (y[nd] == 1).astype(int)
    if yn.sum() == 0 or yn.sum() == len(yn):     # 비구형 라벨이 없으면(정상/결함/제외 3분류) 비구형 판정을 끈다
        accs = [0.0]; thr_ns = 1.01
    else:
        accs = [matthews_corrcoef(yn, (P[nd, 1] >= t).astype(int)) for t in ths]
        thr_ns = float(ths[int(np.argmax(accs))])

    pred = _decide(P, thr_ns, thr_def, thr_ex)
    pd_ = (pred == 2).astype(int)
    lab_ok = y < 3
    rep = dict(
        n=int(len(y)), per_class=counts, n_images=int(n_img),
        thr_exclude=float(thr_ex), thr_defect=float(thr_def), thr_nonspherical=thr_ns,
        confusion_4class=confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
        defect_all=dict(precision=float(precision_score(yd, pd_, zero_division=0)), recall=float(recall_score(yd, pd_, zero_division=0)),
                        f1=float(f1_score(yd, pd_, zero_division=0)),
                        n_true=int(yd.sum()), n_pred=int(pd_.sum()), exclude_leak=int(((y == 3) & (pred == 2)).sum())),
        defect_labeled_only=dict(precision=float(precision_score(yd[lab_ok], pd_[lab_ok], zero_division=0)),
                                 recall=float(recall_score(yd[lab_ok], pd_[lab_ok], zero_division=0))),
        exclude=dict(caught=float((pred[y == 3] == 3).mean()) if (y == 3).any() else None, labeled_lost=float((pred[lab_ok] == 3).mean())),
        nonspherical_vs_round=dict(mcc=float(max(accs)),
                                   balanced_accuracy=float(balanced_accuracy_score(yn, (P[nd, 1] >= thr_ns).astype(int))) if 0 < yn.sum() < len(yn) else None,
                                   accuracy=float(accuracy_score(yn, (P[nd, 1] >= thr_ns).astype(int))) if len(yn) else None),
        cv_folds=int(folds),
        trained_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    samples = d["sample"].fillna("").astype(str).values if "sample" in d.columns else None
    if samples is not None:
        per = {}
        for s in sorted(pd.unique(samples)):
            m_ = samples == s
            judged = m_ & (pred < 3)
            labeled = m_ & lab_ok
            per[s] = dict(n_crops=int(m_.sum()),
                          defect_rate_model=float(100 * (pred[judged] == 2).mean()) if judged.any() else None,
                          defect_rate_label=float(100 * (y[labeled] == 2).mean()) if labeled.any() else None,
                          nonspherical_rate_model=float(100 * (pred[judged] == 1).mean()) if judged.any() else None,
                          nonspherical_rate_label=float(100 * (y[labeled] == 1).mean()) if labeled.any() else None)
        rep["per_sample"] = per
    rep["new_model_cv"] = _metrics(y, pred, samples)
    if compare and os.path.exists(compare):
        try:
            Pc, (tns, tdf, tex) = _proba_with(compare, F)
            rep["current_model"] = _metrics(y, _decide(Pc, tns, tdf, tex), samples)
        except Exception as e:
            log(f"지금 모델과 비교하지 못했습니다 ({e})")
    np.save(os.path.splitext(out)[0] + "_cvprob.npy", np.column_stack([P, y, np.arange(len(y))]))
    tick(0.9, "최종 모델 학습")
    log("최종 모델 학습 중 (전체 라벨)")
    sc, m = fit(np.arange(len(y)))
    W = np.zeros((4, m.coef_.shape[1])); b = np.full(4, -1e6)        # 라벨에 없는 분류는 큰 음수 절편으로 사실상 선택되지 않게
    if len(m.classes_) == 2:                 # sklearn 이진 분류는 가중치 1행만 줌 → 두 행으로 펼침
        W[m.classes_[1]] = m.coef_[0]; b[m.classes_[1]] = m.intercept_[0]; W[m.classes_[0]] = 0; b[m.classes_[0]] = 0
    else:
        W[list(m.classes_)] = m.coef_; b[list(m.classes_)] = m.intercept_
    np.savez(out, mean=sc.mean_, scale=sc.scale_, coef=W, bias=b,
             thr_exclude=thr_ex, thr_defect=thr_def, thr_nonspherical=thr_ns)
    with open(os.path.splitext(out)[0] + "_report.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)
    tick(1.0, "완료")
    log(f"모델 저장: {os.path.basename(out)}")
    return rep
