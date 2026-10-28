"""
가벼운 입자 분류 모델 (PyTorch 없이 numpy만으로 동작)
- 클래스: round(구형 정상) / nonspherical(비구형, 정상) / defect(결함: 판상·균열·박리·조각) / exclude(판정 불가)
  exclude = 겹친 입자, 흐린 입자, 입자 아닌 것 등 사람이 판정에서 뺀 이미지. 프로그램에서는 '판정 불가'로 표시하고 통계에서 제외
- 입력: 128x128 입자 이미지 + 형상 측정값 4개(ECD, 원형도, 종횡비, 볼록도)
- 특징: HOG(윤곽 방향) + LBP(표면 결, 두 반경) + 중심→바깥 밝기 분포 + 밝기·기울기 통계
- 회전·뒤집기 8가지로 본 결과를 평균 (입자가 어느 방향으로 놓여도 같은 판정)
"""
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

CLASSES = ["round", "nonspherical", "defect", "exclude"]

_yy, _xx = np.mgrid[:64, :64]
_R = np.hypot(_yy - 31.5, _xx - 31.5).astype(int)
_RC = np.maximum(np.bincount(_R.ravel(), minlength=46)[:32], 1)


def image_features(im64):
    h = hog(im64, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    lbp = local_binary_pattern(im64, 8, 1, "uniform")
    lh = np.histogram(lbp[16:48, 16:48], bins=10, range=(0, 10), density=True)[0]
    prof = np.bincount(_R.ravel(), im64.ravel().astype(float), minlength=46)[:32] / _RC
    return np.concatenate([h, lh, prof / 255])


def texture_features(im128):
    """회전과 무관한 밝기·질감 통계 (결함은 밝은 얇은 가장자리와 거친 표면이 특징)"""
    c = im128[32:96, 32:96].astype(np.float32)
    lap = cv2.Laplacian(c, cv2.CV_32F)
    gx = cv2.Sobel(c, cv2.CV_32F, 1, 0); gy = cv2.Sobel(c, cv2.CV_32F, 0, 1); gm = np.hypot(gx, gy)
    whole = im128.astype(np.float32)
    lbp2 = local_binary_pattern(im128, 16, 2, "uniform")
    lh2 = np.histogram(lbp2[24:104, 24:104], bins=18, range=(0, 18), density=True)[0]
    return np.concatenate([[c.mean() / 255, c.std() / 255, np.percentile(c, 95) / 255,
                            (whole > 200).mean(), (whole > 230).mean(),
                            lap.var() / 1e4, gm.mean() / 100, gm.std() / 100, np.percentile(gm, 90) / 100], lh2])


def views(im128):
    im = cv2.resize(im128, (64, 64), interpolation=cv2.INTER_AREA)
    out = []
    for k in range(4):
        r = np.rot90(im, k)
        out += [np.ascontiguousarray(r), np.ascontiguousarray(np.fliplr(r))]
    return out


def features(im128, shape4):
    tex = texture_features(im128)
    return np.array([np.concatenate([image_features(v), shape4, tex]) for v in views(im128)])


class ParticleModel:
    def __init__(self, path):
        z = np.load(path)
        self.mean, self.scale, self.coef, self.bias = z["mean"], z["scale"], z["coef"], z["bias"]
        self.thr_exclude = float(z["thr_exclude"])
        self.thr_defect = float(z["thr_defect"])
        self.thr_nonspherical = float(z["thr_nonspherical"])

    def predict(self, crops, shapes):
        """crops: 128x128 uint8 리스트, shapes: (n,4) → (판정 리스트, 결함 확률, 비구형 확률, 판정불가 확률)
        판정: "round" / "nonspherical" / "defect" / "uncertain" """
        labels, p_def, p_ns, p_ex = [], [], [], []
        for c, s in zip(crops, shapes):
            X = (features(c, np.asarray(s, float)) - self.mean) / self.scale
            Z = X @ self.coef.T + self.bias
            Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
            p = P.mean(0)
            if p[3] >= self.thr_exclude:
                lab = "uncertain"
            elif p[2] >= self.thr_defect:
                lab = "defect"
            elif p[1] >= self.thr_nonspherical:
                lab = "nonspherical"
            else:
                lab = "round"
            labels.append(lab); p_def.append(float(p[2])); p_ns.append(float(p[1])); p_ex.append(float(p[3]))
        return labels, p_def, p_ns, p_ex
