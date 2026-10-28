"""
ParticleQC 범용 입자 검사 엔진
- 현미경 사진(SEM/광학 등) → 입자 분할 → 입자별 측정 → 입자 이미지 잘라내기 → AI 판정(normal/abnormal/exclude) → 요약
- 재료·배율·장비가 달라도 되게, 흑연 엔진의 상수를 '설정(settings)'으로 뺐다:
    um_per_px        : 사진 원본 가로 픽셀 1개가 몇 µm인가 (스케일 바로 계산)
    d50_um           : 입자의 대략적인 크기(µm). 내부에서 입자가 약 130 px 지름이 되도록 사진을 키우거나 줄인다
                       → 분할 조건과 AI 입력이 어떤 배율에서도 같은 크기로 맞춰진다
    min_um, max_um   : 입자로 셀 크기 범위
    bright_particles : True = 입자가 배경보다 밝음(SEM SE 등), False = 어두움(광학·투과 등) → 반전해서 처리
    thresh           : 배경보다 얼마나 밝아야 입자로 볼지(0~255 회색조). 대비가 약하면 낮춘다
    np_contrast      : 입자 안쪽이 추정 배경보다 이만큼(회색조) 밝으면 '밝다'. 이보다 어둡고 윤곽까지 흐릿하면 '입자 아님'(테이프 주름 등)
    bar_cut_px       : 사진 아래 정보 바 높이(원본 픽셀). 0이면 자동(어두운 띠 감지)
    agg_contact, agg_sol : 붙은 입자 판정(상대값이라 재료와 무관)
"""
import math, json
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label

TARGET_PX = 130.0            # 내부 작업 해상도에서 d50 입자의 지름(px). 흑연 ×2,000 기준과 같다
CROP_SIZE, MARGIN = 128, 0.20

# 결과 그림의 윤곽선 색(RGB)과 굵기. 노랑·주황은 쓰지 않는다
COLORS = {"normal": (40, 200, 70), "abnormal": (235, 50, 50), "single": (40, 200, 70),
          "agglomerate": (70, 150, 240), "uncertain": (190, 190, 190), "not_particle": (200, 60, 230),
          "manual": (255, 255, 255)}
THICK = {"normal": 3, "abnormal": 3, "single": 2, "agglomerate": 2, "uncertain": 2, "not_particle": 1, "manual": 2}
# 그림 아래 색 설명 띠: (색, 굵기, 한글, 영어). 미리보기(모델 없음)와 검사(모델 있음)
LEGEND_PREVIEW = [
    (COLORS["single"], THICK["single"], "단독 입자 (판정 대상)", "single particle (judged)"),
    (COLORS["agglomerate"], THICK["agglomerate"], "붙은·겹친 입자 (제외)", "touching or overlapping (excluded)"),
    (COLORS["not_particle"], THICK["not_particle"], "입자 아님 (테이프 주름 등, 제외)", "not a particle (excluded)"),
]
LEGEND_MANUAL = (COLORS["manual"], THICK["manual"], "흰색 × = 사람이 뺀 입자 · 흰 점 = 사람이 바꾼 입자", "removed / changed by user")
LEGEND_INSPECT = [
    (COLORS["normal"], THICK["normal"], "normal (정상)", "normal"),
    (COLORS["abnormal"], THICK["abnormal"], "abnormal (불량)", "abnormal"),
    (COLORS["agglomerate"], THICK["agglomerate"], "붙은·겹친 입자 (제외)", "touching or overlapping (excluded)"),
    (COLORS["uncertain"], THICK["uncertain"], "판정 불가 (제외)", "uncertain (excluded)"),
    (COLORS["not_particle"], THICK["not_particle"], "입자 아님 (테이프 주름 등, 제외)", "not a particle (excluded)"),
]

DEFAULTS = dict(um_per_px=0.0655, d50_um=8.5, min_um=2.0, max_um=40.0, bright_particles=True,
                thresh=14, np_contrast=25.0, bar_cut_px=0, agg_contact=0.25, agg_sol=0.88,
                abnormal_max=15.0, use_size=False, d50_min=6.0, d50_max=12.0, span_max=1.0)


def load_settings(path):
    s = dict(DEFAULTS)
    try:
        with open(path, encoding="utf-8") as fh:
            s.update(json.load(fh))
    except Exception:
        pass
    return s


def save_settings(path, s):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=1)


def read_gray(path_or_array):
    if isinstance(path_or_array, np.ndarray):
        a = path_or_array
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a
    data = np.fromfile(path_or_array, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def split_info_bar(img, bar_cut_px=0):
    if bar_cut_px and bar_cut_px > 0:
        return img[:max(1, img.shape[0] - int(bar_cut_px))]
    dark = np.where(img.mean(axis=1) < 30)[0]
    dark = dark[dark > img.shape[0] * 0.8]
    cut = int(dark.min()) if len(dark) else img.shape[0]
    return img[:cut]


def detect_scale_bar(img):
    """사진 아래 정보 바(어두운 띠)에서 가장 긴 밝은 가로선(스케일 바)의 길이(원본 px)를 찾는다. 없으면 None"""
    h = img.shape[0]
    body = split_info_bar(img)
    bar = img[body.shape[0]:, :] if body.shape[0] < h else img[int(h * 0.85):, :]
    best = 0
    for row in (bar > 200).astype(np.int32):
        if row.sum() < 40:
            continue
        runs = np.diff(np.concatenate([[0], row, [0]]))          # 연속된 밝은 구간의 최대 길이
        starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
        if len(starts):
            best = max(best, int((ends - starts).max()))
    return best if best >= 40 else None


def prepare(image, s):
    """원본 → (작업 해상도 이미지, 작업 ppu[px/µm]). 입자가 배경보다 어두우면 반전"""
    raw = read_gray(image)
    img = split_info_bar(raw, s.get("bar_cut_px", 0))
    if not s.get("bright_particles", True):
        img = 255 - img
    ppu_native = 1.0 / float(s["um_per_px"])
    ppu_work = TARGET_PX / float(s["d50_um"])
    scale = float(np.clip(ppu_work / ppu_native, 0.15, 6.0))
    scale = min(scale, 6000.0 / img.shape[1])            # 저배율 사진을 너무 크게 키우지 않는다 (메모리·시간)
    if abs(scale - 1) > 0.01:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return img, ppu_native * scale, scale


def background(img, ppu, d50):
    k = max(9, int(2.5 * d50 * ppu))
    f = min(1.0, 81.0 / k)
    small = cv2.resize(cv2.GaussianBlur(img, (0, 0), 2), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ks = int(k * f) | 1
    bg = cv2.morphologyEx(small, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    bg = cv2.GaussianBlur(bg, (0, 0), 20 * f)
    return cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def segment(img, ppu, d50, thresh, bg=None):
    if bg is None:
        bg = background(img, ppu, d50)
    flat = cv2.GaussianBlur(img.astype(np.float32) - bg.astype(np.float32), (0, 0), 1.5)
    bw = (flat > thresh).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    bw = ndi.binary_fill_holes(bw > 0).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    dist = ndi.distance_transform_edt(bw)
    r = d50 * ppu                                             # d50 입자의 지름(px)
    dist_s = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), max(2.0, 0.06 * r))
    coords = peak_local_max(dist_s, min_distance=max(3, int(0.35 * r)), threshold_abs=0.14 * r, labels=label(bw))
    markers = np.zeros(dist.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    return watershed(-dist, markers, mask=bw)


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def refine_outlines(labels, flat_s, gmag, r_t, core_thr, flap_max, flap_gap, edge_thr=10.0):
    """입자 윤곽 다듬기 (라벨 번호는 그대로)
    (1) 날개: 밝은 몸통에 붙은 어둡고(flap_max 미만) 바깥 윤곽이 흐릿한(edge_thr 미만) 부분이 몸통 밖으로 튀어나와 있으면 잘라냄
        (테이프 주름 등. 몸통의 볼록 껍질 안쪽에 있는 어두운 부분은 입자의 그늘진 쪽이라 자르지 않음)
    (2) 꼬리: 폭 2·r_t px 보다 가는 돌기는 몸통에서 r_t px 넘게 뻗은 부분을 잘라냄 (모서리는 그대로)"""
    k_t = _disk(r_t)
    out = np.zeros_like(labels)
    H, W = labels.shape
    for rp in regionprops(labels):
        minr, minc, maxr, maxc = rp.bbox
        p = 2 * r_t + 2
        y0, x0, y1, x1 = max(0, minr - p), max(0, minc - p), min(H, maxr + p), min(W, maxc + p)
        m = (labels[y0:y1, x0:x1] == rp.label).astype(np.uint8)
        keep = m
        f = flat_s[y0:y1, x0:x1]
        core = ((f >= core_thr) & (m > 0)).astype(np.uint8)
        core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        core = ndi.binary_fill_holes(core).astype(np.uint8) & m
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, _disk(max(2, r_t - 1)))
        n, cc, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        if n > 1:
            body = (cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
            if body.sum() >= 0.5 * m.sum():
                cand = cv2.dilate(body, k_t) & m
                flap = (m > 0) & (cand == 0)
                if flap.sum() >= 0.04 * m.sum():
                    rim = (m > 0) & (cv2.erode(m, np.ones((3, 3), np.uint8)) == 0) & flap
                    fe = float(gmag[y0:y1, x0:x1][rim].mean()) if rim.any() else 99.0
                    fm, cm = float(f[flap].mean()), float(f[body > 0].mean())
                    bc, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    hull = np.zeros_like(body)
                    cv2.fillPoly(hull, [cv2.convexHull(max(bc, key=cv2.contourArea))], 1)
                    in_hull = float((flap & (hull > 0)).sum()) / max(1, int(flap.sum()))
                    if fm < flap_max and fe < edge_thr and cm - fm >= flap_gap and in_hull < 0.3:
                        n2, cc2, st2, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
                        if n2 > 1:
                            k1 = (cc2 == 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
                            kc, _ = cv2.findContours(k1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            ha = cv2.contourArea(cv2.convexHull(max(kc, key=cv2.contourArea)))
                            if ha > 0 and k1.sum() / ha >= 0.90:
                                keep = k1
        op = cv2.morphologyEx(keep, cv2.MORPH_OPEN, k_t)
        n, cc, st, _ = cv2.connectedComponentsWithStats(op, connectivity=8)
        if n > 1:
            keep = cv2.dilate((cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8), k_t) & keep
        out[y0:y1, x0:x1][keep > 0] = rp.label
    return out


# 겹쳐서 하나로 묶인 입자: normal(또는 학습 전 '단독 입자')로 판정됐어도 아래 둘 중 하나면 '붙은 입자(파랑)'로 뺀다
#   (1) 8자·땅콩 모양: 윤곽 양쪽에 잘록한 곳(목)이 마주 보고 있음
#   (2) 크고 한쪽이 깊게 파임: 다른 사진 입자보다 1.25배 이상 크고 윤곽에 깊은 홈이 있음 (한 입자가 다른 입자에 반쯤 겹친 모양)
#   (흑연 사진 기준: 한 윤곽에 묶인 겹친 입자 10개 중 7개를 잡고, 다른 정상 입자 214개 중 3개가 같이 빠짐)
#   (흑연 라벨 2,418개 교차검증: 사람이 정상으로 본 입자는 2.0%가 빠짐). AI가 abnormal 로 본 입자에는 적용하지 않음
#   (가운데가 갈라진 불량 입자도 양쪽이 잘록하기 때문). 위에 얹혀 윤곽에 홈이 없는 겹침은 exclude 로 라벨링해 학습시킨다.
NECK_D1, NECK_D2 = 0.055, 0.045   # 가장 깊은 / 두 번째 홈의 깊이 ÷ 입자 지름
NECK_WIDTH = 0.75                 # 두 홈 사이 거리 ÷ 입자 지름 (이보다 좁으면 '목')
NECK_ANGLE = 110                  # 두 홈을 입자 중심에서 본 각도 (반대쪽)
BIG_NOTCH_D1 = 0.08               # 큰 입자의 깊은 홈 기준
BIG_NOTCH_SIZE = 1.25             # 사진 입자 크기(중앙값) 대비


def notch_info(mask):
    """mask(uint8, 입자 하나)의 윤곽 홈: (가장 깊은 홈, 두 번째 홈, 두 홈 사이 각도, 두 홈 사이 거리) — 깊이·거리는 지름 대비"""
    m = cv2.copyMakeBorder(mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    area = float(m.sum())
    if area < 50:
        return 0.0, 0.0, 0.0, 1.0
    diam = math.sqrt(4 * area / math.pi)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 10:
        return 0.0, 0.0, 0.0, 1.0
    try:
        dfc = cv2.convexityDefects(c, cv2.convexHull(c, returnPoints=False))
    except cv2.error:
        return 0.0, 0.0, 0.0, 1.0
    if dfc is None or len(dfc) == 0:
        return 0.0, 0.0, 0.0, 1.0
    dfc = dfc.reshape(-1, 4)
    depth = dfc[:, 3] / 256.0 / diam
    o = np.argsort(-depth)
    if len(o) < 2:
        return float(depth[o[0]]), 0.0, 0.0, 1.0
    p1, p2 = c[dfc[o[0], 2], 0].astype(float), c[dfc[o[1], 2], 0].astype(float)
    M = cv2.moments(m, True)
    cen = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    v1, v2 = p1 - cen, p2 - cen
    ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1))))
    return float(depth[o[0]]), float(depth[o[1]]), ang, float(np.linalg.norm(p1 - p2)) / diam


def looks_overlapped(d1, d2, ang, neck_w, ecd_rel):
    two_necks = d1 >= NECK_D1 and d2 >= NECK_D2 and ang >= NECK_ANGLE and neck_w <= NECK_WIDTH
    big_notch = d1 >= BIG_NOTCH_D1 and ecd_rel >= BIG_NOTCH_SIZE
    return two_necks or big_notch


def contact_fraction(labels):
    shared = {}
    for dy, dx in [(0, 1), (1, 0)]:
        a = labels[:labels.shape[0] - dy, :labels.shape[1] - dx]
        b = labels[dy:, dx:]
        m = (a > 0) & (b > 0) & (a != b)
        if not m.any():
            continue
        pr = np.sort(np.stack([a[m], b[m]], 1), 1)
        u, n = np.unique(pr, axis=0, return_counts=True)
        for (x, y), k in zip(u, n):
            shared[(x, y)] = shared.get((x, y), 0) + k
    area = np.bincount(labels.ravel())
    diam = np.sqrt(4 * area / np.pi)
    frac = {}
    for (x, y), k in shared.items():
        f = (k / 2) / max(1.0, min(diam[x], diam[y]))
        frac[x] = max(frac.get(x, 0), f); frac[y] = max(frac.get(y, 0), f)
    return frac


def crop_particle(img, rp):
    minr, minc, maxr, maxc = rp.bbox
    cy, cx = rp.centroid
    side = int(max(maxr - minr, maxc - minc) * (1 + 2 * MARGIN))
    y0, x0 = int(cy - side / 2), int(cx - side / 2)
    pad = cv2.copyMakeBorder(img, side, side, side, side, cv2.BORDER_REFLECT)
    crop = pad[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def analyze(image, s, classify_fn=None, want_crops=False, return_parts=False):
    """
    image: 파일 경로 또는 numpy 배열. s: settings dict
    classify_fn(crops, shapes) -> (labels, p_abnormal, p_unused, p_uncertain); labels는 "round"/"defect"/"uncertain"
      (내부 모델 이름: round=normal, defect=abnormal). None이면 AI 없이 '단독 입자'만 표시
    반환: overlay(RGB), DataFrame[id,cx,cy,ecd_um,circularity,aspect_ratio,solidity,contact,judge,...],
          want_crops=True 이면 잘라낸 입자 그림, return_parts=True 이면 그림을 다시 그릴 부품 dict 도 함께
    """
    img, ppu, scale = prepare(image, s)
    d50 = float(s["d50_um"])
    bg = background(img, ppu, d50)
    labels = segment(img, ppu, d50, float(s["thresh"]), bg)
    flat_raw = img.astype(np.float32) - bg.astype(np.float32)
    gb = cv2.GaussianBlur(img, (0, 0), 1.0)
    gmag = np.hypot(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
    cfrac = contact_fraction(labels)          # 맞닿음은 다듬기 전 분할로 판단
    # 윤곽 다듬기: 가는 꼬리(폭 ≈ d50의 8% 미만)와 테이프 주름 날개를 잘라 윤곽선이 입자 가장자리를 따르게 한다
    npc = float(s["np_contrast"])
    labels = refine_outlines(labels, cv2.GaussianBlur(flat_raw, (0, 0), 1.5), gmag,
                             r_t=max(2, int(round(0.038 * d50 * ppu))), core_thr=npc - 1.0, flap_max=0.72 * npc, flap_gap=0.8 * npc)
    ring_k = max(5, int(0.07 * d50 * ppu) | 1)
    h, w = img.shape
    rows, crops, contours, rejected = [], [], [], []
    for rp in regionprops(labels):
        minr, minc, maxr, maxc = rp.bbox
        if minr == 0 or minc == 0 or maxr >= h or maxc >= w:
            continue
        ecd = math.sqrt(4 * rp.area / math.pi) / ppu
        if ecd < float(s["min_um"]) or ecd > float(s["max_um"]):
            continue
        mask = (labels[minr:maxr, minc:maxc] == rp.label).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        perim = cv2.arcLength(cv2.approxPolyDP(cnt, 1.0, True), True)
        circ = min(1.0, 4 * math.pi * rp.area / perim ** 2) if perim > 0 else 0
        ar = rp.axis_major_length / rp.axis_minor_length if rp.axis_minor_length > 0 else 99
        cnt = cnt + np.array([[[minc, minr]]])
        pad = ring_k
        y0, x0 = max(0, minr - pad), max(0, minc - pad)
        sub = (labels[y0:maxr + pad, x0:maxc + pad] == rp.label).astype(np.uint8)
        ring = cv2.dilate(sub, np.ones((ring_k, ring_k), np.uint8)) - sub
        fsub = flat_raw[y0:maxr + pad, x0:maxc + pad]
        inside = float(fsub[sub > 0].mean())                   # 추정 배경보다 얼마나 밝은가
        edge_px = cv2.dilate(mask, np.ones((3, 3), np.uint8)) - cv2.erode(mask, np.ones((3, 3), np.uint8))
        edge = float(gmag[minr:maxr, minc:maxc][edge_px > 0].mean()) if edge_px.any() else 0.0   # 윤곽 선명도
        # 입자 아님(테이프 주름 조각): 배경과 비슷하게 어둡고(inside < np_contrast) 윤곽이 흐릿한(edge < 10) 것.
        # 모양이 아주 불규칙한 것은 어둡거나 흐릿할 때만 테이프로 본다. 밝고 윤곽이 또렷하면 모양과 관계없이 입자로 넘긴다
        # → 붙은 입자 무리는 아래 접촉 필터(파랑)가, 애매한 것은 AI(판정 불가)가 거른다. 실제 입자가 '입자 아님'이 되지 않도록.
        dim, soft = inside < float(s["np_contrast"]), edge < 10.0
        if (dim and soft) or (circ < 0.5 and rp.solidity < 0.85 and (dim or soft)):
            rejected.append(cnt)
            continue
        contact = cfrac.get(rp.label, 0) > float(s["agg_contact"]) or rp.solidity < float(s["agg_sol"]) or ecd > 3.0 * d50
        rows.append(dict(id=len(rows) + 1, cx=int(rp.centroid[1] / scale), cy=int(rp.centroid[0] / scale),
                         ecd_um=round(ecd, 2), circularity=round(circ, 3), aspect_ratio=round(ar, 3),
                         solidity=round(rp.solidity, 3), contact=bool(contact),
                         ))
        rows[-1]["notch1"], rows[-1]["notch2"], rows[-1]["notch_ang"], rows[-1]["neck_w"] = (round(v, 3) for v in notch_info(mask))
        crops.append(crop_particle(img, rp))
        contours.append(cnt)
    df = pd.DataFrame(rows)
    df.attrs["n_not_particle"] = len(rejected)
    parts = dict(img=img, contours=contours, rejected=rejected, scale=scale, ppu=ppu)
    if df.empty:
        overlay = render(parts, df)
        if want_crops:
            return overlay, df, []
        return (overlay, df, parts) if return_parts else (overlay, df)
    df["judge"] = np.where(df["contact"], "agglomerate", "single")
    iso = ~df["contact"].values
    if classify_fn is not None and iso.any():
        shapes = df.loc[iso, ["ecd_um", "circularity", "aspect_ratio", "solidity"]].values.astype(float)
        labs, p_ab, _, p_ex = classify_fn([c for c, k in zip(crops, iso) if k], shapes)
        name = {"round": "normal", "nonspherical": "normal", "defect": "abnormal", "uncertain": "uncertain"}
        df.loc[iso, "judge"] = [name.get(l, l) for l in labs]
        df.loc[iso, "p_abnormal"] = np.round(p_ab, 3)
        df.loc[iso, "p_uncertain"] = np.round(p_ex, 3)
    # 겹쳐서 하나로 묶인 입자(8자 모양 / 크고 한쪽이 깊게 파임)로 normal·단독 판정된 것은 '붙은 입자'로 뺀다
    df["judge_ai"] = df["judge"]                      # 겹침 규칙 전의 판정 (사람이 되돌릴 때 씀)
    ecd_rel = df["ecd_um"] / max(float(df["ecd_um"].median()), 1e-6)
    over = np.array([looks_overlapped(a, b, c, d, e) for a, b, c, d, e in
                     zip(df["notch1"], df["notch2"], df["notch_ang"], df["neck_w"], ecd_rel)])
    df["overlap"] = over & ~df["contact"].values & df["judge"].isin(["normal", "single"]).values
    df.loc[df["overlap"], "judge"] = "agglomerate"
    df["judge_auto"] = df["judge"]                    # 자동 판정 (사람이 고치기 전)
    df["edited"] = False                              # 사람이 판정을 바꿨는지
    df["from_np"] = False                             # '입자 아님(보라)'이던 것을 사람이 입자로 넣었는지
    overlay = render(parts, df)
    if want_crops:
        return overlay, df, crops
    return (overlay, df, parts) if return_parts else (overlay, df)


def render(parts, df):
    """판정 결과 그림 (화면 표시용으로 가로 2600 px 이하로 줄임).
    초록 = normal(또는 학습 전 단독 입자), 빨강 = abnormal, 파랑 = 붙은·겹친 입자, 회색 = 판정 불가,
    흰색 × = 사람이 뺀 입자, 흰 점 = 사람이 판정을 바꾼 입자, 보라 얇은 선 = 입자 아님"""
    overlay = cv2.cvtColor(parts["img"], cv2.COLOR_GRAY2RGB)
    for cnt in parts["rejected"]:
        cv2.drawContours(overlay, [cnt], -1, COLORS["not_particle"], THICK["not_particle"])
    if df is not None and not df.empty:
        r = max(6, int(0.012 * overlay.shape[1]))
        edited = df["edited"] if "edited" in df else [False] * len(df)
        for cnt, j, ed in zip(parts["contours"], df["judge"], edited):
            cv2.drawContours(overlay, [cnt], -1, COLORS[j], THICK[j])
            if j == "manual" or bool(ed):
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    if j == "manual":                     # 사람이 뺀 입자: 가운데 × 표시
                        cv2.line(overlay, (cx - r, cy - r), (cx + r, cy + r), COLORS["manual"], 3)
                        cv2.line(overlay, (cx - r, cy + r), (cx + r, cy - r), COLORS["manual"], 3)
                    else:                                 # 사람이 판정을 바꾼 입자: 가운데 흰 점
                        cv2.circle(overlay, (cx, cy), max(4, r // 2), (255, 255, 255), -1)
                        cv2.circle(overlay, (cx, cy), max(4, r // 2), (40, 40, 40), 1)
    if overlay.shape[1] > 2600:                           # 화면 표시용으로 줄임
        f = 2600.0 / overlay.shape[1]
        overlay = cv2.resize(overlay, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return overlay


JUDGED = ("normal", "abnormal", "single")
KOREAN = {"normal": "normal(정상)", "abnormal": "abnormal(불량)", "single": "단독 입자", "agglomerate": "붙은·겹친 입자(제외)",
          "uncertain": "판정 불가(제외)", "manual": "사람이 뺌(흰 ×)", "not_particle": "입자 아님"}


def _hit(contours, x, y):
    """점(x, y)를 품은 윤곽 중 가장 작은 것의 번호"""
    best = None
    for i, cnt in enumerate(contours):
        if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
            a = cv2.contourArea(cnt)
            if best is None or a < best[1]:
                best = (i, a)
    return None if best is None else best[0]


def _row_from_contour(cnt, ppu, scale, next_id):
    """'입자 아님'으로 빠졌던 윤곽을 입자 한 줄로 측정 (좌표는 원본 사진 기준)"""
    x, y, w, h = cv2.boundingRect(cnt)
    m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.drawContours(m, [cnt - np.array([[[x - 1, y - 1]]])], -1, 1, -1)
    rp = regionprops(m)[0]
    perim = cv2.arcLength(cv2.approxPolyDP(cnt, 1.0, True), True)
    circ = min(1.0, 4 * math.pi * rp.area / perim ** 2) if perim > 0 else 0
    ar = rp.axis_major_length / rp.axis_minor_length if rp.axis_minor_length > 0 else 99
    d1, d2, ang, nw = notch_info(m)
    return dict(id=next_id, cx=int((x - 1 + rp.centroid[1]) / scale), cy=int((y - 1 + rp.centroid[0]) / scale),
                ecd_um=round(math.sqrt(4 * rp.area / math.pi) / ppu, 2), circularity=round(circ, 3), aspect_ratio=round(ar, 3),
                solidity=round(rp.solidity, 3), contact=False,
                notch1=round(d1, 3), notch2=round(d2, 3), notch_ang=round(ang, 3), neck_w=round(nw, 3),
                judge="not_particle", judge_ai="not_particle", overlap=False, judge_auto="not_particle", edited=True, from_np=True)


def _kind(j):
    return "normal" if j in ("normal", "single") else j


def set_judge(parts, df, x, y, target, has_model=True):
    """화면에서 클릭한 점(작업 해상도 좌표)의 입자를 사람이 고른 판정으로 바꾼다.
    target: "normal"(초록) / "abnormal"(빨강) / "agglomerate"(파랑, 제외) / "uncertain"(회색, 제외) / "manual"(흰 ×, 제외) / "auto"(자동 판정으로)
    같은 것을 고른 채 다시 누르면 자동 판정으로 되돌린다. 보라(입자 아님)를 normal·abnormal 등으로 고르면 입자로 넣는다."""
    i = _hit(parts["contours"], x, y)
    if i is None:
        k = _hit(parts["rejected"], x, y)
        if k is None:
            return "입자 윤곽 안쪽을 클릭해 주세요."
        if target in ("auto", "manual"):
            return "보라색(입자 아님)은 이미 통계에서 빠져 있습니다. 입자로 넣으려면 normal 이나 abnormal 을 고른 뒤 누르세요."
        cnt = parts["rejected"].pop(k)
        row = _row_from_contour(cnt, parts["ppu"], parts["scale"], int(df["id"].max()) + 1)
        if "image" in df.columns:
            row["image"] = df["image"].iloc[0]
        df.loc[len(df)] = pd.Series(row)
        parts["contours"].append(cnt)
        df.attrs["n_not_particle"] = max(0, int(df.attrs.get("n_not_particle", 0)) - 1)
        i = len(df) - 1
    elif target == "auto" or (bool(df.at[i, "edited"]) and _kind(df.at[i, "judge"]) == target):
        if bool(df.at[i, "from_np"]):                     # 원래 '입자 아님'이던 것 → 다시 보라로
            parts["rejected"].append(parts["contours"].pop(i))
            df.drop(index=i, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.attrs["n_not_particle"] = int(df.attrs.get("n_not_particle", 0)) + 1
            return "입자 아님(보라)으로 되돌렸습니다."
        df.at[i, "judge"] = df.at[i, "judge_auto"]
        df.at[i, "edited"] = False
        return f"자동 판정으로 되돌렸습니다 → {KOREAN.get(df.at[i, 'judge'], df.at[i, 'judge'])}"
    new = ("normal" if has_model else "single") if target == "normal" else target
    df.at[i, "judge"] = new
    df.at[i, "edited"] = bool(new != df.at[i, "judge_auto"])
    return f"{KOREAN.get(new, new)}(으)로 바꿨습니다. 같은 것을 고른 채 다시 누르면 되돌립니다."


def reset_edits(parts, df):
    """사람이 고친 것을 모두 취소 (입자로 넣었던 보라 윤곽은 다시 '입자 아님'으로)"""
    if df is None or df.empty:
        return
    for i in sorted(df.index[df["from_np"].astype(bool)], reverse=True):
        parts["rejected"].append(parts["contours"].pop(i))
        df.drop(index=i, inplace=True)
        df.attrs["n_not_particle"] = int(df.attrs.get("n_not_particle", 0)) + 1
    df.reset_index(drop=True, inplace=True)
    df["judge"] = df["judge_auto"]
    df["edited"] = False

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def homogeneity(ks, ns):
    ks, ns = np.asarray(ks, float), np.asarray(ns, float)
    m = ns > 0
    ks, ns = ks[m], ns[m]
    if len(ns) < 3 or ks.sum() == 0 or ks.sum() == ns.sum():
        return float("nan"), float("nan")
    p = ks.sum() / ns.sum()
    chi = float((((ks - ns * p) ** 2) / (ns * p * (1 - p))).sum())
    dof = len(ns) - 1
    from scipy.stats import chi2
    return chi / dof, float(chi2.sf(chi, dof))


def summarize(df, s):
    d = df["ecd_um"].values
    d10, d50, d90 = np.percentile(d, [10, 50, 90])
    span = (d90 - d10) / d50
    judged = df[df["judge"].isin(["normal", "abnormal"])]
    n1 = max(len(judged), 1)
    k_ab = int((judged["judge"] == "abnormal").sum())
    ab = 100 * k_ab / n1
    lo, hi = wilson(k_ab, len(judged))
    checks = {}
    if s.get("use_size", False):
        checks[f"D50 {s['d50_min']}~{s['d50_max']} um"] = float(s["d50_min"]) <= d50 <= float(s["d50_max"])
        checks[f"분포 폭 ≤ {s['span_max']}"] = span <= float(s["span_max"])
    checks[f"abnormal 입자 ≤ {s['abnormal_max']}%"] = ab <= float(s["abnormal_max"])
    out = {
        "입자 수": len(df), "D10 (um)": round(d10, 2), "D50 (um)": round(d50, 2), "D90 (um)": round(d90, 2),
        "분포 폭 (D90-D10)/D50": round(span, 2),
        "판정 대상 (단독 입자)": len(judged),
        "평균 원형도": round(float(judged["circularity"].mean()), 3) if len(judged) else float("nan"),
        "평균 종횡비": round(float(judged["aspect_ratio"].mean()), 2) if len(judged) else float("nan"),
        "abnormal 입자 (%)": round(ab, 1),
        "abnormal 95% 신뢰구간": f"{lo:.1f} ~ {hi:.1f}%  (abnormal {k_ab}개 / 판정 {len(judged)}개)" if len(judged) else "",
        "abnormal 입자 수": k_ab,
        "응집·접촉 입자 (%, 판정 제외)": round(100 * (df["judge"] == "agglomerate").mean(), 1),
        "판정 불가 (%, 겹침·흐림 등)": round(100 * (df["judge"] == "uncertain").mean(), 1),
        "겹친 입자로 뺀 수 (자동)": int(df["overlap"].sum()) if "overlap" in df else 0,
        "사람이 직접 뺀 수": int((df["judge"] == "manual").sum()),
        "사람이 판정을 바꾼 수": int(df["edited"].astype(bool).sum()) if "edited" in df else 0,
        "입자 아님 (제외)": int(df.attrs.get("n_not_particle", 0)),
    }
    return out, checks, all(checks.values())
