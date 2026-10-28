"""
새 ×2,000 사진 → 단독 입자 잘라내기 → 라벨링 시트 만들기
사용법: python ingest_new_photos.py <사진 폴더> <출력 폴더> <접두어>
  - 출력: <출력 폴더>/crops/<접두어>_<사진>_pNNNN.png, crops.csv, 시트 sheet_XX.png
  - 접촉 필터를 통과한 단독 입자만 자름 (응집·접촉 입자는 제외)
"""
import sys, os, glob, math
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graphite_engine as ge
from skimage.measure import regionprops

src, out, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(out, "crops"), exist_ok=True)
rows = []
for f in sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.tif"))):
    raw = ge.read_gray(f)
    ppu = ge.px_per_um(2000, raw.shape[1]); img = ge.split_info_bar(raw)
    scale = ge.REF_WIDTH / raw.shape[1]
    if abs(scale - 1) > 0.01:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC); ppu = ge.MAG_SETTINGS[2000][1]
    bg = ge.background(img, ppu); labels = ge.segment(img, ppu, bg)
    flat = img.astype(np.float32) - bg.astype(np.float32); cfrac = ge.contact_fraction(labels)
    ring_k = max(5, int(0.6 * ppu) | 1); h, w = img.shape
    stem = os.path.splitext(os.path.basename(f))[0]; n = 0
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
        name = f"{prefix}_{stem}_p{rp.label:04d}.png"
        ok, buf = cv2.imencode(".png", ge.crop_particle(img, rp)); buf.tofile(os.path.join(out, "crops", name))
        rows.append(dict(file=name, image=os.path.basename(f), cx=int(rp.centroid[1]), cy=int(rp.centroid[0]), ecd_um=round(ecd, 2),
                         circularity=round(circ, 3), aspect_ratio=round(ar, 3), solidity=round(rp.solidity, 3)))
        n += 1
    print(f"{os.path.basename(f)}: {n} single crops", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "crops.csv"), index=False, encoding="utf-8-sig")
# 라벨링 시트
C, R, T = 10, 6, 150
for s in range(0, len(df), C * R):
    sheet = np.full((R * (T + 4), C * (T + 4), 3), 255, np.uint8)
    for j, name in enumerate(df.file[s:s + C * R]):
        im = cv2.imdecode(np.fromfile(os.path.join(out, "crops", name), np.uint8), 0)
        im = cv2.cvtColor(cv2.resize(im, (T, T), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        r, c = divmod(j, C); y, x = r * (T + 4), c * (T + 4); sheet[y:y + T, x:x + T] = im; t = str(s + j)
        cv2.rectangle(sheet, (x, y), (x + 8 + 10 * len(t), y + 17), (0, 0, 0), -1)
        cv2.putText(sheet, t, (x + 3, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(out, f"sheet_{s // (C * R):02d}.png"), sheet)
print("total", len(df), "sheets", (len(df) + C * R - 1) // (C * R))
