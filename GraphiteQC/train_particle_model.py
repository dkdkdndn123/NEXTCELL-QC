"""
입자 4분류 모델 학습: round / nonspherical / defect / exclude(판정 불가)
사용법: python train_particle_model.py <labeling 폴더> <출력 npz> [--cache <특징 캐시 npz>]
- labeling 폴더 구조: round/, nonspherical/, defect/, exclude/ + crops.csv
  (정상/결함/제외 3분류로 쓰려면 정상을 round/ 에 넣고 nonspherical/ 은 비워 두면 됨 — 비구형 판정이 꺼진 모델이 만들어짐)
  crops.csv 필수 열: file, image, ecd_um, circularity, aspect_ratio, solidity  (선택: sample)
- 사진 단위 5겹 교차검증으로 성능을 낸 뒤, 전체 데이터로 최종 모델 학습
- 기준값: 판정불가(thr_exclude)와 결함(thr_defect)은 결함 F1 최대(±0.01) 범위에서 예측 결함 수가 실제와 가장 가까운 값, 비구형(thr_nonspherical)은 매튜스 상관계수(MCC) 최대 — 구형이 훨씬 많아도 비구형 비율이 라벨과 맞게 잡힘
- sample 열이 있으면 시료별(원본/손상 등) 결함 비율을 교차검증 예측과 라벨로 비교해 출력
"""
import sys, os, json, hashlib
import numpy as np, pandas as pd, cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from particle_model import features, CLASSES

LAB, OUT = sys.argv[1], sys.argv[2]
CACHE = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else None
meta = pd.read_csv(os.path.join(LAB, "crops.csv"))
rows = []
for ci, cls in enumerate(CLASSES):
    folder = os.path.join(LAB, cls)
    if not os.path.isdir(folder):          # 비어 있거나 없는 분류(예: 비구형을 쓰지 않을 때)도 허용
        continue
    for f in sorted(os.listdir(folder)):
        if f.endswith(".png"):
            rows.append((folder + os.sep + f, f, ci))
d = pd.DataFrame(rows, columns=["path", "file", "y"]).merge(meta, on="file")
y = d.y.values; g = d.image.values
shapes = d[["ecd_um", "circularity", "aspect_ratio", "solidity"]].values.astype(float)

# 특징 추출 (캐시가 있으면 파일 내용 해시로 재사용)
def file_key(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()
cache = {}
if CACHE and os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True); cache = dict(zip(z["keys"], z["feats"]))
keys = [file_key(p) for p in d.path]
F = []
for k, p, s in zip(keys, d.path, shapes):
    if k in cache and cache[k].shape[1] == len(features(np.zeros((128, 128), np.uint8), s)[0]):
        F.append(cache[k])
    else:
        F.append(features(cv2.imdecode(np.fromfile(p, np.uint8), 0), s)); cache[k] = F[-1]
if CACHE:
    np.savez(CACHE, keys=np.array(list(cache.keys())), feats=np.array(list(cache.values()), dtype=object))
counts = dict(zip(CLASSES, np.bincount(y, minlength=4).tolist()))
print("samples per class:", counts, "| images:", len(set(g)), flush=True)


def fit(idx, C=0.03):
    X = np.vstack([F[i] for i in idx]); Y = np.repeat(y[idx], 8)
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=8000, class_weight="balanced").fit(sc.transform(X), Y)
    return sc, m


def proba4(m, X):
    """라벨에 없는 분류가 있어도 항상 4열(round/nonspherical/defect/exclude) 확률로 맞춘다"""
    P = np.zeros((len(X), 4)); P[:, list(m.classes_)] = m.predict_proba(X)
    return P


def coef4(m):
    """4분류 가중치. 라벨에 없는 분류는 큰 음수 절편으로 사실상 선택되지 않게 한다"""
    W = np.zeros((4, m.coef_.shape[1])); b = np.full(4, -1e6)
    if len(m.classes_) == 2:                 # sklearn 이진 분류는 가중치 1행만 줌 → 두 행으로 펼침
        W[m.classes_[1]] = m.coef_[0]; b[m.classes_[1]] = m.intercept_[0]; W[m.classes_[0]] = 0; b[m.classes_[0]] = 0
    else:
        W[list(m.classes_)] = m.coef_; b[list(m.classes_)] = m.intercept_
    return W, b


P = np.zeros((len(y), 4))
for k, (tr, te) in enumerate(GroupKFold(5).split(F, y, g)):
    sc, m = fit(tr)
    for i in te:
        P[i] = proba4(m, sc.transform(F[i])).mean(0)
    print(f"fold {k + 1}/5 done", flush=True)

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


def decide(P):
    out = np.zeros(len(P), int)
    out[P[:, 1] >= thr_ns] = 1
    out[P[:, 2] >= thr_def] = 2
    out[P[:, 3] >= thr_ex] = 3
    return out


pred = decide(P)
pd_ = (pred == 2).astype(int)
lab = y < 3
rep = dict(
    n=int(len(y)), per_class=counts, n_images=int(len(set(g))),
    thr_exclude=float(thr_ex), thr_defect=float(thr_def), thr_nonspherical=thr_ns,
    confusion_4class=confusion_matrix(y, pred).tolist(),
    defect_all=dict(precision=float(precision_score(yd, pd_)), recall=float(recall_score(yd, pd_)), f1=float(f1_score(yd, pd_)),
                    n_true=int(yd.sum()), n_pred=int(pd_.sum()), exclude_leak=int(((y == 3) & (pred == 2)).sum())),
    defect_labeled_only=dict(precision=float(precision_score(yd[lab], pd_[lab])), recall=float(recall_score(yd[lab], pd_[lab]))),
    exclude=dict(caught=float((pred[y == 3] == 3).mean()), labeled_lost=float((pred[lab] == 3).mean())),
    nonspherical_vs_round=dict(mcc=float(max(accs)),
                               balanced_accuracy=float(balanced_accuracy_score(yn, (P[nd, 1] >= thr_ns).astype(int))) if 0 < yn.sum() < len(yn) else None,
                               accuracy=float(accuracy_score(yn, (P[nd, 1] >= thr_ns).astype(int)))),
)
if "sample" in d.columns:
    per = {}
    for s in sorted(d["sample"].unique()):
        m_ = (d["sample"] == s).values
        judged = m_ & (pred < 3)
        labeled = m_ & lab
        per[s] = dict(n_crops=int(m_.sum()),
                      defect_rate_model=float(100 * (pred[judged] == 2).mean()) if judged.any() else None,
                      defect_rate_label=float(100 * (y[labeled] == 2).mean()) if labeled.any() else None,
                      nonspherical_rate_model=float(100 * (pred[judged] == 1).mean()) if judged.any() else None,
                      nonspherical_rate_label=float(100 * (y[labeled] == 1).mean()) if labeled.any() else None)
    rep["per_sample"] = per
np.save(os.path.splitext(OUT)[0] + "_cvprob.npy", np.column_stack([P, y, np.arange(len(y))]))
print(json.dumps(rep, indent=1, ensure_ascii=False))
sc, m = fit(np.arange(len(y)))
W4, b4 = coef4(m)
np.savez(OUT, mean=sc.mean_, scale=sc.scale_, coef=W4, bias=b4,
         thr_exclude=thr_ex, thr_defect=thr_def, thr_nonspherical=thr_ns)
json.dump(rep, open(os.path.splitext(OUT)[0] + "_report.json", "w"), indent=1, ensure_ascii=False)
print("saved", OUT)
