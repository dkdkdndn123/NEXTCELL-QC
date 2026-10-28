"""
결과 그림 아래에 색 설명 띠(색 견본 + 한글 설명)를 붙인다.
그림을 그릴 때 쓴 색표를 그대로 받아서 그리므로, 화면에 보이는 색과 설명이 어긋나지 않는다.
한글 글꼴(맑은 고딕 등)을 못 찾으면 영어 설명으로 대신한다.
"""
import os
import numpy as np

_FONTS = {}


def _font(size):
    if size in _FONTS:
        return _FONTS[size]
    from PIL import ImageFont
    windir = os.environ.get("WINDIR", r"C:\Windows")
    cands = [os.path.join(windir, "Fonts", n) for n in ("malgunbd.ttf", "malgun.ttf", "gulim.ttc", "batang.ttc")]
    cands += ["/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
              "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/Library/Fonts/AppleGothic.ttf"]
    f = None
    for p in cands:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                break
            except Exception:
                pass
    _FONTS[size] = f
    return f


def add_legend(overlay, items, credit=None):
    """overlay: RGB uint8 배열. items: [(rgb, 선 굵기, 한글 설명, 영어 설명), ...]
    사진 배경과 비슷한 회색 띠를 아래에 붙여, 그림 속 윤곽선과 같은 색·굵기의 견본과 설명을 적는다.
    credit: 띠 오른쪽 아래에 작게 적는 프로그램·만든 사람 표시 (결과 그림을 캡처해도 출처가 남도록)"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return overlay
    h, w = overlay.shape[:2]
    fs = max(14, int(round(w / 60)))                  # 글자 크기 (가로 2576 px → 43 px)
    font = _font(fs)
    use_ko = font is not None
    if font is None:
        try:
            font = ImageFont.load_default(size=fs)
        except Exception:
            font = ImageFont.load_default()
    pad = int(fs * 0.7)
    sw, sh = int(fs * 1.9), int(fs * 1.05)            # 색 견본(작은 입자 모양 타원)
    gap = int(fs * 1.3)
    measure = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    rows, row, x = [], [], pad
    for rgb, t, ko, en in items:
        text = ko if use_ko else en
        iw = sw + int(fs * 0.45) + int(measure.textlength(text, font=font))
        if row and x + iw > w - pad:
            rows.append(row)
            row, x = [], pad
        row.append((rgb, t, text, x))
        x += iw + gap
    if row:
        rows.append(row)
    row_h = int(fs * 1.75)
    cfont, cw = None, 0
    if credit:
        cfont = _font(max(11, int(fs * 0.62))) if use_ko else font
        cw = int(measure.textlength(credit, font=cfont))
        # 마지막 줄 오른쪽에 자리가 없으면 한 줄 더
        tail = rows[-1][-1] if rows and rows[-1] else None
        used = (tail[3] + sw + int(fs * 0.45) + int(measure.textlength(tail[2], font=font))) if tail else pad
        credit_new_row = used + gap + cw > w - pad
    else:
        credit_new_row = False
    n_rows = len(rows) + (1 if credit_new_row else 0)
    band = Image.new("RGB", (w, pad + n_rows * row_h), (88, 88, 88))
    d = ImageDraw.Draw(band)
    for r, items_row in enumerate(rows):
        top = pad // 2 + r * row_h
        for rgb, t, text, x in items_row:
            lw = max(2, int(round(t * fs / 18)))      # 그림에서 굵은 선은 견본도 굵게, 얇은 선은 얇게
            y = top + (row_h - sh) // 2
            d.ellipse([x, y, x + sw, y + sh], fill=(150, 150, 150), outline=tuple(int(c) for c in rgb), width=lw)
            d.text((x + sw + int(fs * 0.45), top + row_h // 2), text, font=font, fill=(250, 250, 250), anchor="lm")
    if credit:
        top = pad // 2 + (n_rows - 1) * row_h
        d.text((w - pad, top + row_h // 2), credit, font=cfont, fill=(205, 205, 205), anchor="rm")
    return np.vstack([overlay, np.asarray(band, dtype=np.uint8)])
