"""
흑연 입자 SEM 품질검사 엔진 (시연 프로그램용)
- 입자 분할 → 입자별 측정 → 입자 이미지 잘라내기 → AI 판정 → 결과 요약
- AI 판정 함수는 바깥에서 넣어줌 (classify_fn), 없으면 규칙 기반으로 대체
"""
import math
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label

MAG_SETTINGS = {500: (100.0, 3.81), 2000: (30.0, 15.27)}   # 배율: (스케일바 um, px/um) — 가로 2576 px 기준
REF_WIDTH = 2576                                            # 보정값을 구한 사진의 가로 픽셀 수


def px_per_um(mag, width):
    """저장 해상도가 달라도 같은 시야이므로 가로 픽셀 수에 비례해 환산값을 조정"""
    return MAG_SETTINGS[mag][1] * width / REF_WIDTH
THRESH = 14
MIN_ECD_UM, MAX_ECD_UM = 2.0, 40.0
FINE_UM = 5.0
CROP_SIZE, MARGIN = 128, 0.20

# 결과 그림의 윤곽선 색(RGB)과 굵기. 노랑·주황은 쓰지 않는다 ('입자 아님'으로 오해하기 쉬워서).
#   정상 입자는 모양(둥근 것·길쭉한 것)과 관계없이 모두 초록. 길쭉한 비율은 표의 '비구형 %'로 본다.
COLORS = {"round": (40, 200, 70), "nonspherical": (40, 200, 70), "defect": (235, 50, 50),
          "agglomerate": (70, 150, 240), "uncertain": (190, 190, 190), "not_particle": (200, 60, 230),
          "manual": (255, 255, 255)}
THICK = {"round": 3, "nonspherical": 3, "defect": 3, "agglomerate": 2, "uncertain": 2, "not_particle": 1, "manual": 2}
# 그림 아래 색 설명 띠에 쓰는 항목: (색, 굵기, 한글, 영어)
LEGEND_ITEMS = [
    (COLORS["round"], THICK["round"], "정상 입자 (둥근 것·길쭉한 것 모두)", "normal (round or elongated)"),
    (COLORS["defect"], THICK["defect"], "결함 (판상·균열·깨진 조각)", "defect"),
    (COLORS["agglomerate"], THICK["agglomerate"], "붙은·겹친 입자 (판정 제외)", "touching or overlapping (excluded)"),
    (COLORS["uncertain"], THICK["uncertain"], "판정 불가 (제외)", "uncertain (excluded)"),
    (COLORS["not_particle"], THICK["not_particle"], "테이프 주름 (입자 아님, 제외)", "tape wrinkle (not a particle)"),
    (COLORS["manual"], THICK["manual"], "흰색 × = 사람이 뺀 입자 · 흰 점 = 사람이 바꾼 입자", "removed / changed by user"),
]


def read_gray(path_or_array):
    if isinstance(path_or_array, np.ndarray):
        a = path_or_array
        return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a
    data = np.fromfile(path_or_array, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def split_info_bar(img):
    dark = np.where(img.mean(axis=1) < 30)[0]
    dark = dark[dark > img.shape[0] * 0.8]
    cut = int(dark.min()) if len(dark) else img.shape[0]
    return img[:cut]


def background(img, ppu):
    k = int(21 * ppu)
    f = min(1.0, 81.0 / k)
    small = cv2.resize(cv2.GaussianBlur(img, (0, 0), 2), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ks = int(k * f) | 1
    bg = cv2.morphologyEx(small, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    bg = cv2.GaussianBlur(bg, (0, 0), 20 * f)
    return cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def segment(img, ppu, bg=None):
    if bg is None:
        bg = background(img, ppu)
    flat = cv2.GaussianBlur(img.astype(np.float32) - bg.astype(np.float32), (0, 0), 1.5)
    bw = (flat > THRESH).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    bw = ndi.binary_fill_holes(bw > 0).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    dist = ndi.distance_transform_edt(bw)
    dist_s = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), max(2.0, 0.5 * ppu))
    coords = peak_local_max(dist_s, min_distance=max(3, int(3.0 * ppu)),
                            threshold_abs=1.2 * ppu, labels=label(bw))
    markers = np.zeros(dist.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    return watershed(-dist, markers, mask=bw)


# 입자가 아닌 영역(카본 테이프 주름·균열·찢어진 부분) 거르기
#   테이프 주름 조각은 ① 배경과 비슷하게 어둡고 ② 윤곽이 흐릿하다. 둘 다 해당해야 '입자 아님'.
#   밝거나 윤곽이 또렷한 것은 모양이 아무리 불규칙해도 입자(또는 붙은 입자 무리)로 넘긴다
#   → 뒤의 응집·접촉 필터(파랑)나 AI(판정 불가 = 회색)가 거른다. 실제 입자가 '입자 아님'이 되는 일이 없도록.
NP_INSIDE = 25.0            # 입자 안쪽이 추정 배경보다 이만큼(회색조) 밝으면 '밝다'. 테이프 주름 띠는 15~18 수준
NP_EDGE = 10.0              # 윤곽선의 밝기 기울기(선명도). 테이프 주름은 2~9, 납작하게 누운 어두운 입자도 20 이상
NP_CIRC, NP_SOL = 0.5, 0.85  # 모양이 아주 불규칙한 기준 (어둡거나 흐릿한 것과 같이 쓸 때만)


def is_not_particle(contrast, circ, sol, inside=None, edge=None):
    """입자 아님(테이프 주름·찢어진 조각) 판정
    - 어둡고(inside < NP_INSIDE) 윤곽이 흐릿하면(edge < NP_EDGE) 테이프 주름
    - 모양이 아주 불규칙한 것은 어둡거나 흐릿할 때만 테이프로 본다
    - 밝고 윤곽이 또렷한 것은 모양과 관계없이 입자로 넘긴다 (붙은 입자 무리 → 파랑, 애매한 것 → AI가 회색)
    예전의 '안쪽-바깥 대비 < 20', '불규칙하면 무조건 제외' 규칙은 빽빽한 자리나 납작하게 누운 입자를 빼 버려 없앴다"""
    if inside is None:
        inside = contrast
    dim = inside < NP_INSIDE
    soft = edge is None or edge < NP_EDGE
    if dim and soft:
        return True
    if circ < NP_CIRC and sol < NP_SOL and (dim or soft):
        return True
    return False


# 윤곽 다듬기: 자동 분할에서 생기는 '꼬리'와 '날개'를 잘라 윤곽선이 입자 가장자리를 따르게 한다
#   꼬리 = 입자 옆 작은 부스러기·테이프 결이 가늘게 이어져 윤곽이 고리·갈고리처럼 튀어나온 것
#   날개 = 밝은 입자에 테이프 주름(어둡고 흐릿한 띠)이 붙어 윤곽이 입자 밖으로 넓게 번진 것
TAIL_UM = 0.33              # 폭이 이 값의 2배(0.66 µm, ×2,000에서 10 px)보다 가는 돌기는 몸통에서 뻗은 부분을 잘라낸다
FLAP_CORE = 24.0            # 입자 몸통으로 볼 밝기(배경 대비 회색조)
FLAP_MAX = 18.0             # 날개의 평균 밝기가 이보다 어둡고(테이프 주름 수준), 바깥 윤곽이 NP_EDGE 보다 흐릿하고,
FLAP_GAP = 20.0             #   몸통이 날개보다 이만큼 이상 밝을 때만 자른다 → 납작하게 누운 어두운 입자의 몸은 자르지 않는다


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def refine_outlines(labels, flat_s, gmag, ppu):
    """입자마다 (1) 테이프 날개 (2) 가는 꼬리를 잘라 낸 라벨 영상을 돌려준다. 라벨 번호는 그대로."""
    r_t = max(2, int(round(TAIL_UM * ppu)))
    k_t = _disk(r_t)
    out = np.zeros_like(labels)
    H, W = labels.shape
    for rp in regionprops(labels):
        minr, minc, maxr, maxc = rp.bbox
        p = 2 * r_t + 2
        y0, x0, y1, x1 = max(0, minr - p), max(0, minc - p), min(H, maxr + p), min(W, maxc + p)
        m = (labels[y0:y1, x0:x1] == rp.label).astype(np.uint8)
        keep = m
        # (1) 날개: 밝은 몸통 + 둘레 r_t px(가장자리 테)만 남기고, 나머지가 어둡고 흐릿한 테이프면 잘라냄
        f = flat_s[y0:y1, x0:x1]
        core = ((f >= FLAP_CORE) & (m > 0)).astype(np.uint8)
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
                    # 날개는 몸통 바깥으로 튀어나와 있어야 한다. 몸통의 볼록 껍질 안쪽에 있는 어두운 부분은
                    # 입자의 그늘진 쪽(몸의 일부)이므로 자르지 않는다 (자르면 U·V 모양이 되어 결함처럼 보인다)
                    bc, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    hull = np.zeros_like(body)
                    cv2.fillPoly(hull, [cv2.convexHull(max(bc, key=cv2.contourArea))], 1)
                    in_hull = float((flap & (hull > 0)).sum()) / max(1, int(flap.sum()))
                    if fm < FLAP_MAX and fe < NP_EDGE and cm - fm >= FLAP_GAP and in_hull < 0.3:
                        n2, cc2, st2, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
                        if n2 > 1:
                            k1 = (cc2 == 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
                            kc, _ = cv2.findContours(k1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            ha = cv2.contourArea(cv2.convexHull(max(kc, key=cv2.contourArea)))
                            if ha > 0 and k1.sum() / ha >= 0.90:     # 자른 뒤 모양이 볼록한 입자 모양일 때만 채택
                                keep = k1
        # (2) 꼬리: 열림 연산으로 가는 돌기를 떼어 낸 몸통 둘레 r_t px 안의 원래 윤곽만 되살림 (모서리는 그대로, 꼬리만 잘림)
        op = cv2.morphologyEx(keep, cv2.MORPH_OPEN, k_t)
        n, cc, st, _ = cv2.connectedComponentsWithStats(op, connectivity=8)
        if n > 1:
            keep = cv2.dilate((cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8), k_t) & keep
        out[y0:y1, x0:x1][keep > 0] = rp.label
    return out


# 겹쳐서 하나로 묶인 입자: 정상(초록)으로 판정됐어도 아래 둘 중 하나면 '붙은 입자(파랑)'로 뺀다
#   (1) 8자·땅콩 모양: 윤곽 양쪽에 잘록한 곳(목)이 마주 보고 있음
#   (2) 크고 한쪽이 깊게 파임: 다른 사진 입자보다 1.25배 이상 크고 윤곽에 깊은 홈이 있음 (한 입자가 다른 입자에 반쯤 겹친 모양)
#   보낸 사진의 초록 입자 229개를 눈으로 본 겹침 15개 중 7개를 잡고, 나머지 214개 중 3개가 같이 빠짐.
#   사람 라벨 2,418개 교차검증: 사람이 정상으로 본 입자는 2.0%가 빠짐 (결함 비율 원본 11.2→11.6%, 사람 10.7%). AI가 결함으로 본 입자에는 적용하지 않음
#   (가운데가 갈라진 결함 입자도 양쪽이 잘록하기 때문). 남은 겹침은 화면에서 클릭해 판정을 바꾼다(앱의 '결과 고치기').
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


# 맞붙거나 뭉친 입자(응집·겹침): 시료를 펼칠 때 생기는 현상이라 불량 판정에서 제외
AGG_SOL = 0.88              # 분리되지 않은 덩어리는 볼록도가 낮음
AGG_CONTACT = 0.25          # 이웃 입자와 맞닿은 경계 길이 / 작은 쪽 입자 지름이 이보다 크면 맞붙은 것


def contact_fraction(labels):
    """입자별로 이웃 입자와 맞닿은 경계 길이 비율의 최댓값"""
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


def rule_label(ecd, circ, ar, sol):
    """규칙 기반 (AI 없을 때): 원형도·볼록도가 매우 낮으면 결함, 길쭉하거나 각지면 비구형"""
    if circ < 0.75 or sol < 0.93:
        return "defect"
    if circ < 0.85 or ar > 1.6:
        return "nonspherical"
    return "round"


def crop_particle(img, rp):
    minr, minc, maxr, maxc = rp.bbox
    cy, cx = rp.centroid
    side = int(max(maxr - minr, maxc - minc) * (1 + 2 * MARGIN))
    y0, x0 = int(cy - side / 2), int(cx - side / 2)
    pad = cv2.copyMakeBorder(img, side, side, side, side, cv2.BORDER_REFLECT)
    crop = pad[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def analyze(image, mag=2000, classify_fn=None, return_parts=False):
    """
    image: 파일 경로 또는 numpy 배열(RGB/그레이)
    classify_fn(crops, shapes) -> (labels, p_defect, p_nonspherical, p_uncertain)
                 labels: "round" / "nonspherical" / "defect" / "uncertain" 리스트. crops: 128x128 uint8 리스트,
                 shapes: (n,4) [ecd_um, circularity, aspect_ratio, solidity]. None이면 규칙 기반 사용
    반환: overlay(RGB), 입자 DataFrame  (return_parts=True 이면 그림을 다시 그릴 부품 dict 도 함께)
    """
    raw = read_gray(image)
    ppu = px_per_um(mag, raw.shape[1])
    img = split_info_bar(raw)
    # AI 모델 입력과 분할 조건을 학습 때와 맞추기 위해 기준 해상도로 맞춘 뒤 분석
    scale = REF_WIDTH / raw.shape[1]
    if abs(scale - 1) > 0.01:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        ppu = MAG_SETTINGS[mag][1]
    bg = background(img, ppu)
    labels = segment(img, ppu, bg)
    flat_raw = img.astype(np.float32) - bg.astype(np.float32)
    gb = cv2.GaussianBlur(img, (0, 0), 1.0)
    gmag = np.hypot(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))   # 윤곽 선명도용
    cfrac = contact_fraction(labels)          # 맞닿음은 다듬기 전 분할로 판단 (꼬리·날개로 이어진 것도 접촉으로 남김)
    if ppu >= 8:                              # ×2,000급에서만 윤곽 다듬기 (×500은 입자가 작아 그대로)
        labels = refine_outlines(labels, cv2.GaussianBlur(flat_raw, (0, 0), 1.5), gmag, ppu)
    ring_k = max(5, int(0.6 * ppu) | 1)
    h, w = img.shape
    rows, crops, contours = [], [], []
    rejected = []
    for rp in regionprops(labels):
        minr, minc, maxr, maxc = rp.bbox
        if minr == 0 or minc == 0 or maxr >= h or maxc >= w:
            continue
        ecd = math.sqrt(4 * rp.area / math.pi) / ppu
        if ecd < MIN_ECD_UM or ecd > MAX_ECD_UM:
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
        inside = float(fsub[sub > 0].mean())
        contrast = float(inside - fsub[ring > 0].mean()) if ring.any() else 99.0
        edge_px = cv2.dilate(mask, np.ones((3, 3), np.uint8)) - cv2.erode(mask, np.ones((3, 3), np.uint8))
        edge = float(gmag[minr:maxr, minc:maxc][edge_px > 0].mean()) if edge_px.any() else 0.0
        if is_not_particle(contrast, circ, rp.solidity, inside, edge):
            rejected.append(cnt)
            continue
        contact = cfrac.get(rp.label, 0) > AGG_CONTACT or rp.solidity < AGG_SOL or ecd > 25
        rows.append(dict(id=len(rows) + 1, cx=int(rp.centroid[1]), cy=int(rp.centroid[0]),
                         ecd_um=round(ecd, 2), circularity=round(circ, 3), aspect_ratio=round(ar, 3),
                         solidity=round(rp.solidity, 3), contact=bool(contact),
                         rule="agglomerate" if contact else rule_label(ecd, circ, ar, rp.solidity)))
        rows[-1]["notch1"], rows[-1]["notch2"], rows[-1]["notch_ang"], rows[-1]["neck_w"] = (round(v, 3) for v in notch_info(mask))
        crops.append(crop_particle(img, rp))
        contours.append(cnt)
    df = pd.DataFrame(rows)
    df.attrs["n_not_particle"] = len(rejected)
    parts = dict(img=img, contours=contours, rejected=rejected, mag=mag, ppu=ppu)
    if df.empty:
        ov = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return (ov, df, parts) if return_parts else (ov, df)
    df["judge"] = df["rule"]
    iso = ~df["contact"].values
    if classify_fn is not None and iso.any():
        # AI는 혼자 떨어진 입자만 판정: 구형 / 비구형(정상) / 결함(판상·균열·박리·조각) / 판정불가(겹침·흐림 등)
        shapes = df.loc[iso, ["ecd_um", "circularity", "aspect_ratio", "solidity"]].values.astype(float)
        labs, p_def, p_ns, p_ex = classify_fn([c for c, k in zip(crops, iso) if k], shapes)
        df.loc[iso, "judge"] = list(labs)
        df.loc[iso, "p_defect"] = np.round(p_def, 3)
        df.loc[iso, "p_nonspherical"] = np.round(p_ns, 3)
        df.loc[iso, "p_uncertain"] = np.round(p_ex, 3)
    # 겹쳐서 하나로 묶인 입자(8자 모양 / 크고 한쪽이 깊게 파임)로 정상 판정된 것은 '붙은 입자'로 뺀다
    df["judge_ai"] = df["judge"]                      # 겹침 규칙 전의 판정 (사람이 되돌릴 때 씀)
    ecd_rel = df["ecd_um"] / max(float(df["ecd_um"].median()), 1e-6)
    over = np.array([looks_overlapped(a, b, c, d, e) for a, b, c, d, e in
                     zip(df["notch1"], df["notch2"], df["notch_ang"], df["neck_w"], ecd_rel)])
    df["overlap"] = over & ~df["contact"].values & df["judge"].isin(["round", "nonspherical"]).values
    df.loc[df["overlap"], "judge"] = "agglomerate"
    df["judge_auto"] = df["judge"]                    # 자동 판정 (사람이 고치기 전)
    df["edited"] = False                              # 사람이 판정을 바꿨는지
    df["from_np"] = False                             # '입자 아님(보라)'이던 것을 사람이 입자로 넣었는지
    overlay = render(parts, df)
    return (overlay, df, parts) if return_parts else (overlay, df)


def render(parts, df):
    """판정 결과 그림. 초록 = 정상(둥근 것·길쭉한 것), 빨강 = 결함, 파랑 = 응집·접촉, 회색 = 판정 불가,
    흰색 × = 사람이 뺀 입자, 흰 점 = 사람이 판정을 바꾼 입자, 보라 얇은 선 = 테이프 주름 (파랑·회색·흰색·보라는 통계 제외)"""
    overlay = cv2.cvtColor(parts["img"], cv2.COLOR_GRAY2RGB)
    for cnt in parts["rejected"]:
        cv2.drawContours(overlay, [cnt], -1, COLORS["not_particle"], THICK["not_particle"])
    if df is None or df.empty:
        return overlay
    mag = parts["mag"]
    r = max(6, int(0.012 * overlay.shape[1]))
    edited = df["edited"] if "edited" in df else [False] * len(df)
    for cnt, j, cx, cy, ed in zip(parts["contours"], df["judge"], df["cx"], df["cy"], edited):
        t = THICK[j] if mag == 2000 else max(1, THICK[j] - 1)
        cv2.drawContours(overlay, [cnt], -1, COLORS[j], t)
        if j == "manual":                                 # 사람이 뺀 입자: 가운데 × 표시
            cv2.line(overlay, (int(cx) - r, int(cy) - r), (int(cx) + r, int(cy) + r), COLORS["manual"], 3)
            cv2.line(overlay, (int(cx) - r, int(cy) + r), (int(cx) + r, int(cy) - r), COLORS["manual"], 3)
        elif bool(ed):                                    # 사람이 판정을 바꾼 입자: 가운데 흰 점
            cv2.circle(overlay, (int(cx), int(cy)), max(4, r // 2), (255, 255, 255), -1)
            cv2.circle(overlay, (int(cx), int(cy)), max(4, r // 2), (40, 40, 40), 1)
    return overlay


JUDGED = ("round", "nonspherical", "defect")


def _hit(contours, x, y):
    """점(x, y)를 품은 윤곽 중 가장 작은 것의 번호"""
    best = None
    for i, cnt in enumerate(contours):
        if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
            a = cv2.contourArea(cnt)
            if best is None or a < best[1]:
                best = (i, a)
    return None if best is None else best[0]


def _normal_kind(df, i):
    """'정상'으로 바꿀 때 둥근/길쭉한 구분: AI 판정이 있으면 그것, 없으면 모양으로"""
    ai = df.at[i, "judge_ai"]
    if ai in ("round", "nonspherical"):
        return ai
    return "nonspherical" if (df.at[i, "circularity"] < 0.85 or df.at[i, "aspect_ratio"] > 1.6) else "round"


def _row_from_contour(cnt, ppu, next_id):
    """'입자 아님'으로 빠졌던 윤곽을 입자 한 줄로 측정"""
    x, y, w, h = cv2.boundingRect(cnt)
    m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.drawContours(m, [cnt - np.array([[[x - 1, y - 1]]])], -1, 1, -1)
    rp = regionprops(m)[0]
    perim = cv2.arcLength(cv2.approxPolyDP(cnt, 1.0, True), True)
    circ = min(1.0, 4 * math.pi * rp.area / perim ** 2) if perim > 0 else 0
    ar = rp.axis_major_length / rp.axis_minor_length if rp.axis_minor_length > 0 else 99
    d1, d2, ang, nw = notch_info(m)
    return dict(id=next_id, cx=int(x - 1 + rp.centroid[1]), cy=int(y - 1 + rp.centroid[0]),
                ecd_um=round(math.sqrt(4 * rp.area / math.pi) / ppu, 2), circularity=round(circ, 3), aspect_ratio=round(ar, 3),
                solidity=round(rp.solidity, 3), contact=False, rule="not_particle",
                notch1=round(d1, 3), notch2=round(d2, 3), notch_ang=round(ang, 3), neck_w=round(nw, 3),
                judge="not_particle", judge_ai="not_particle", overlap=False, judge_auto="not_particle", edited=True, from_np=True)


def _kind(j):
    return "normal" if j in ("round", "nonspherical") else j


def set_judge(parts, df, x, y, target):
    """화면에서 클릭한 점(작업 해상도 좌표)의 입자를 사람이 고른 판정으로 바꾼다.
    target: "normal"(초록) / "defect"(빨강) / "agglomerate"(파랑, 제외) / "uncertain"(회색, 제외) / "manual"(흰 ×, 제외) / "auto"(자동 판정으로)
    같은 것을 고른 채 다시 누르면 자동 판정으로 되돌린다. 보라(입자 아님)를 정상·결함 등으로 고르면 입자로 넣는다."""
    i = _hit(parts["contours"], x, y)
    if i is None:
        k = _hit(parts["rejected"], x, y)
        if k is None:
            return "입자 윤곽 안쪽을 클릭해 주세요."
        if target in ("auto", "manual"):
            return "보라색(입자 아님)은 이미 통계에서 빠져 있습니다. 입자로 넣으려면 '정상'이나 '결함'을 고른 뒤 누르세요."
        cnt = parts["rejected"].pop(k)
        row = _row_from_contour(cnt, parts["ppu"], int(df["id"].max()) + 1)
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
    new = _normal_kind(df, i) if target == "normal" else target
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


KOREAN = {"round": "정상(둥근 입자)", "nonspherical": "정상(길쭉한 입자)", "defect": "결함", "agglomerate": "붙은·겹친 입자(제외)",
          "uncertain": "판정 불가(제외)", "manual": "사람이 뺌(흰 ×)", "not_particle": "입자 아님"}


def wilson(k, n, z=1.96):
    """비율 k/n 의 95% 신뢰구간 (Wilson). 입자 수가 적은 사진 한 장의 결함률이 얼마나 흔들리는지 보여 준다."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # 0개·전부일 때는 구간 끝을 정확히 0%·100%로 둔다 (부동소수점 오차로 비율보다 살짝 커지거나 작아지면 그래프가 멈춤)
    lo = 0.0 if k <= 0 else max(0.0, c - h)
    hi = 1.0 if k >= n else min(1.0, c + h)
    return 100 * lo, 100 * hi


def homogeneity(ks, ns):
    """사진별 결함 수 ks / 판정 입자 수 ns 가 '같은 비율 + 셈 오차'로 설명되는지 (카이제곱 동질성 검정).
    반환: (dispersion, p값). dispersion ≈ 1 이면 셈 오차뿐, 2 이상이면 사진(촬영 위치)마다 실제로 다름."""
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


def summarize(df, d50_range=(6.0, 12.0), span_max=1.0, defect_max=15.0, circ_min=0.85):
    d = df["ecd_um"].values
    d10, d50, d90 = np.percentile(d, [10, 50, 90])
    span = (d90 - d10) / d50
    single = df[df["judge"].isin(["round", "nonspherical", "defect"])]   # 판정된 단독 입자만 형상 통계
    n1 = max(len(single), 1)
    k_def = int((single["judge"] == "defect").sum())
    defect = 100 * k_def / n1
    ci_lo, ci_hi = wilson(k_def, len(single))
    nonsph = 100 * (single["judge"] == "nonspherical").sum() / n1
    agg = 100 * (df["judge"] == "agglomerate").mean()
    unc = 100 * (df["judge"] == "uncertain").mean()
    circ_single = float(single["circularity"].mean()) if len(single) else float("nan")
    checks = {
        f"D50 {d50_range[0]}~{d50_range[1]} um": d50_range[0] <= d50 <= d50_range[1],
        f"분포 폭 ≤ {span_max}": span <= span_max,
    }
    if circ_min is not None:   # 구형화 제품(천연흑연)용 규격 항목. 둥글지 않은 제품(인조흑연 등)은 None 으로 끈다
        checks[f"평균 원형도 ≥ {circ_min} (단독 입자)"] = circ_single >= circ_min
    checks[f"결함 입자 ≤ {defect_max}%"] = defect <= defect_max
    s = {
        "입자 수": len(df), "D10 (um)": round(d10, 2), "D50 (um)": round(d50, 2), "D90 (um)": round(d90, 2),
        "분포 폭 (D90-D10)/D50": round(span, 2), "미분 <5um (%)": round(100 * (d < FINE_UM).mean(), 1),
        "── 구형화 지표 (단독 입자 기준) ──": "",
        "판정 대상 (단독 입자)": len(single),
        "평균 원형도": round(circ_single, 3), "평균 종횡비": round(float(single["aspect_ratio"].mean()), 2) if len(single) else float("nan"),
        "비구형 입자 (%, 정상)": round(nonsph, 1),
        "── 결함 ──": "",
        "결함 입자 (%, 판상·균열·조각)": round(defect, 1),
        "결함 비율 95% 신뢰구간": f"{ci_lo:.1f} ~ {ci_hi:.1f}%  (결함 {k_def}개 / 판정 {len(single)}개)" if len(single) else "",
        "결함 입자 수": k_def,
        "── 제외 ──": "",
        "응집·접촉 입자 (%, 판정 제외)": round(agg, 1),
        "판정 불가 (%, 겹침·흐림 등)": round(unc, 1),
        "겹친 입자로 뺀 수 (자동)": int(df["overlap"].sum()) if "overlap" in df else 0,
        "사람이 직접 뺀 수": int((df["judge"] == "manual").sum()),
        "사람이 판정을 바꾼 수": int(df["edited"].astype(bool).sum()) if "edited" in df else 0,
        "입자 아님 (테이프 주름, 제외)": int(df.attrs.get("n_not_particle", 0)),
    }
    return s, checks, all(checks.values())
