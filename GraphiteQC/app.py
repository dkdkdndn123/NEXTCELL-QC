"""
흑연 입자 AI 품질검사 · NEXT CELL
바탕화면 아이콘으로 실행하면 이 파일이 실행되고, 브라우저에 검사 화면이 열립니다.
인터넷 없이 내 컴퓨터에서만 동작합니다.

판정 구조
  - 입도(D10/D50/D90, 분포 폭): 찾은 모든 입자로 계산
  - 구형화 지표(평균 원형도, 비구형 비율): 혼자 떨어진 입자 중 AI가 판정한 것만
  - 결함 비율(판상·균열·박리·조각): 같은 입자들 중 AI가 결함으로 판정한 비율
  - 응집·접촉, 판정 불가(겹침·흐림), 입자 아님(테이프)은 형상 통계에서 제외
  - 사진 한 장은 판정 입자가 수십 개뿐이라 결함률이 ±5%p 안팎 흔들린다(셈 오차). 그래서 여러 장을 합산해 판정하고,
    합산 결함률의 95% 신뢰구간과 사진 간 동질성(셈 오차로 설명되는지)을 함께 표시한다
"""
import os, sys, tempfile, threading, webbrowser
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import socket
import gradio as gr

VERSION = "v1.0.3 (2026-10-29)"
TEAM = "NEXT CELL"
MEMBERS = "윤성빈(팀장) · 김태욱 · 공진우 · 허동현"
SCHOOL = "순천제일대학교 에너지소재화공과"
CREDIT = f"GraphiteQC {VERSION} · {TEAM} · {SCHOOL}"
ABOUT = f"""### 만든 사람
**{TEAM}** · {SCHOOL}  
{MEMBERS}

이 팀이 직접 한 일
- 구형화 천연흑연 시료 준비와 SEM 촬영 (×2,000 · 20 kV · SE · 코팅, 원래 시료와 4분 동안 갈아 손상시킨 시료)
- 입자 2,418개를 사진 36장에서 하나씩 분류(구형 · 비구형 · 결함 · 판정 불가)하고, 결함(판상·균열·박리·깨진 조각)의 기준을 정함
- 그 분류로 AI를 학습시키고, 학습에 쓰지 않은 사진으로 사람 판정과 비교해 검증 (원래 흑연 결함 비율: AI 11.2% · 사람 10.7%)
- 합격 기준(D50, 분포 폭, 원형도, 결함 비율)과 사진 장수에 따른 95% 신뢰구간 · '판정 보류' 규칙을 정함

버전 {VERSION} · © 2026 {TEAM}. MIT 라이선스로 공개 (프로그램 폴더의 LICENSE 파일). 고쳐 쓰거나 나눠 줄 때는 이 저작권 표시를 남겨 주세요.
"""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graphite_engine as ge
import color_legend as cl

MODEL_PATH = os.path.join(HERE, "particle_model.npz")
PORT = 7860

# ---------------------------------------------------------------- AI 모델 (프로그램에 포함)
classify = None
try:
    from particle_model import ParticleModel
    _model = ParticleModel(MODEL_PATH)
    classify = _model.predict
    model_msg = "AI 판정 모델 준비 완료"
except Exception as e:
    model_msg = f"AI 모델을 불러오지 못했습니다 ({e}). 규칙 기반으로 판정합니다."
print(model_msg)

LEGEND = ("**그림 색** (결과 그림 아래 회색 띠에도 같은 설명이 있습니다): "
          "초록 = 정상 입자(둥근 것·길쭉한 것 모두) · 빨강 = 결함(판상·균열·박리·깨진 조각) · "
          "파랑 = 붙은 입자(겹친 입자 포함, 판정 제외) · 회색 = 판정 불가(흐림 등, 제외) · 보라 얇은 선 = 테이프 주름(입자 아님, 제외) · "
          "흰색 × = 사람이 뺀 입자 · 흰 점 = 사람이 판정을 바꾼 입자. 노란색·주황색 선은 쓰지 않습니다.")


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def fmt(v):
    if isinstance(v, float):
        return "" if v != v else f"{v:g}"
    return str(v)


# ---------------------------------------------------------------- 검사
EDIT_W = 1600          # '결과 고치기' 그림의 표시 가로 크기(px). 클릭 좌표는 이 비율로 되돌려 쓴다


def analyze_files(paths, mag, use_ai):
    """사진마다 분석해 (이름, 입자 표, 그림 부품)을 모은다. 사람이 고친 뒤 다시 그리고 다시 계산할 때 이것을 쓴다"""
    entries, seen = [], {}
    for p in paths:
        img = read_image(p)
        _, df, parts = ge.analyze(img, mag, classify if use_ai else None, return_parts=True)
        name = os.path.basename(p)
        if name in seen:                       # 같은 이름의 사진이 둘이면 구분
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        if not df.empty:
            df.insert(0, "image", name)
        entries.append(dict(name=name, df=df, parts=parts))
    return entries


def entry_overlay(e):
    return cl.add_legend(ge.render(e["parts"], e["df"]), ge.LEGEND_ITEMS, credit=CREDIT)


def summarize_entries(entries, mag, d50_min, d50_max, span_max, circ_min, defect_max, exclude_bad=True, use_circ=True):
    plt.close("all")
    circ_min = circ_min if use_circ else None     # 원형도 규격은 구형화 천연흑연용. 끄면 결함·크기 항목만으로 판정
    use_ai = (mag == 2000) and classify is not None
    gallery, dfs, per_img = [], [], []
    for e in entries:
        overlay = entry_overlay(e)
        name, df = e["name"], e["df"]
        if df.empty:
            gallery.append((overlay, f"{name}: 입자 없음"))
            continue
        dfs.append(df)
        s, _, _ = ge.summarize(df, (d50_min, d50_max), span_max, defect_max, circ_min)
        lo, hi = ge.wilson(s["결함 입자 수"], s["판정 대상 (단독 입자)"])
        per_img.append({"사진": name, "품질": "", "입자 수": s["입자 수"], "테이프 주름": int(df.attrs.get("n_not_particle", 0)),
                        "D50 (um)": s["D50 (um)"], "판정 입자": s["판정 대상 (단독 입자)"],
                        "평균 원형도": s["평균 원형도"], "비구형 (%)": s["비구형 입자 (%, 정상)"],
                        "결함 (개)": s["결함 입자 수"], "결함 (%)": s["결함 입자 (%, 판상·균열·조각)"],
                        "결함 95% 구간": f"{lo:.0f}~{hi:.0f}%" if s["판정 대상 (단독 입자)"] else "",
                        "접촉 (%)": s["응집·접촉 입자 (%, 판정 제외)"], "판정불가 (%)": s["판정 불가 (%, 겹침·흐림 등)"],
                        "겹침 자동 제외": s["겹친 입자로 뺀 수 (자동)"], "사람이 바꿈": s["사람이 판정을 바꾼 수"],
                        "사람이 뺌": s["사람이 직접 뺀 수"]})
        gallery.append((overlay, f"{name} · 결함 {s['결함 입자 (%, 판상·균열·조각)']:g}% · D50 {s['D50 (um)']:g} um"))
    if not dfs:
        return gallery, "입자를 찾지 못했습니다. 배율 선택이 맞는지 확인해 주세요.", None, None, None, None
    # 사진 품질 점검: 테이프 주름·접힘이 큰 사진은 배경 추정이 깨져 입자를 놓치고 테이프 조각을 입자 후보로 잡는다
    #  → 찾은 입자 수가 다른 사진 중앙값의 절반 이하이거나, 후보 중 '입자 아님'이 30% 이상이면 주름 의심
    med = float(np.median([r["입자 수"] for r in per_img]))
    bad = []
    for i, r in enumerate(per_img):
        cand = r["입자 수"] + r["테이프 주름"]
        if (len(per_img) >= 3 and r["입자 수"] < 0.5 * med) or (cand and r["테이프 주름"] / cand >= 0.30):
            bad.append(i)
    note_bad = ""
    if bad:
        names = ", ".join(per_img[i]["사진"] for i in bad)
        if exclude_bad and len(dfs) - len(bad) >= 3:
            for i in bad:
                per_img[i]["품질"] = "⚠ 주름 의심 → 제외"
            dfs = [d for i, d in enumerate(dfs) if i not in bad]
            per_img_used = [r for i, r in enumerate(per_img) if i not in bad]
            note_bad = f"\n\n⚠️ 테이프 주름·접힘이 의심되는 사진 {len(bad)}장을 합산에서 제외했습니다: {names}. 평평한 자리에서 다시 찍어 보충하세요."
        else:
            for i in bad:
                per_img[i]["품질"] = "⚠ 주름 의심"
            per_img_used = per_img
            note_bad = f"\n\n⚠️ 테이프 주름·접힘이 의심되는 사진이 있습니다: {names}. 이런 사진은 입자를 놓치므로 빼고 다시 올리는 것을 권장합니다."
    else:
        per_img_used = per_img
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.attrs["n_not_particle"] = sum(d.attrs.get("n_not_particle", 0) for d in dfs)
    s, checks, ok = ge.summarize(all_df, (d50_min, d50_max), span_max, defect_max, circ_min)
    if use_ai:
        method = "AI (구형 / 비구형 / 결함 / 판정불가)"
    elif mag == 2000:
        method = "규칙 기반 (AI 모델 불러오기 실패)"
    else:
        method = "규칙 기반 (×500은 입도 측정용)"
    n_img = len(dfs)
    ks = [r["결함 (개)"] for r in per_img_used]; ns = [r["판정 입자"] for r in per_img_used]
    n_single, k_def = int(sum(ns)), int(sum(ks))
    lo, hi = ge.wilson(k_def, n_single)
    half = (hi - lo) / 2 if n_single else float("nan")
    disp, p_hom = ge.homogeneity(ks, ns)
    # 결함 비율의 신뢰구간이 기준에 걸치면(다른 항목은 통과) 합격/불합격 대신 '보류' — 사진을 더 찍으면 판정이 뒤집힐 수 있는 상태
    borderline = bool(n_single) and lo <= defect_max <= hi and all(v for k, v in checks.items() if not k.startswith("결함"))
    head = "⚠️ 판정 보류 (사진 추가 필요)" if borderline else ("✅ 합격" if ok else "❌ 불합격")
    verdict = f"## {head}\n사진 {n_img}장 합산 · 판정 방법: **{method}**\n\n"
    verdict += "\n".join(f"- {'✅' if v else '❌'} {k}" for k, v in checks.items())
    verdict += note_bad
    n_man = int((all_df["judge"] == "manual").sum())
    n_ed = int(all_df["edited"].astype(bool).sum()) if "edited" in all_df else 0
    if n_ed:
        verdict += f"\n\n✋ 사람이 판정을 바꾼 입자 {n_ed}개(그중 뺀 것 {n_man}개)를 반영한 결과입니다."
    if n_single:
        verdict += (f"\n\n**결함 비율 {s['결함 입자 (%, 판상·균열·조각)']:g}%** (결함 {k_def}개 / 판정 입자 {n_single}개 · "
                    f"95% 신뢰구간 {lo:.1f}~{hi:.1f}%, ±{half:.1f}%p)")
        # 셈 오차: 입자 n개에서 비율 p의 흔들림은 ±1.96·√(p(1-p)/n). 목표 ±3%p 에 필요한 입자 수 → 사진 수
        p_ = max(k_def, 1) / n_single
        need_particles = p_ * (1 - p_) / (0.03 / 1.96) ** 2
        need_imgs = int(np.ceil(need_particles / max(n_single / n_img, 1)))
        if lo <= defect_max <= hi:
            verdict += (f"\n\n⚠️ 결함 비율의 신뢰구간이 기준 {defect_max:g}%에 걸쳐 있어 사진을 더 찍으면 판정이 바뀔 수 있습니다. "
                        f"같은 시료를 {max(need_imgs - n_img, 3)}장 이상 더 찍어 함께 올리세요.")
        elif n_img < 10:
            verdict += (f"\n\n⚠️ 사진 {n_img}장(판정 입자 {n_single}개)은 결함 비율이 ±{half:.1f}%p 흔들립니다. "
                        f"±3%p 안으로 줄이려면 같은 시료를 약 {need_imgs}장 찍어 함께 올리세요.")
    if p_hom == p_hom and n_img >= 5:
        if p_hom < 0.05:
            verdict += (f"\n\nℹ️ 사진별 결함률 차이가 입자 수로 설명되는 범위보다 큽니다(사진 간 산포 {disp:.1f}배, p={p_hom:.3f}). "
                        "시료가 고르지 않거나(갈림·응집이 한쪽에 몰림) 촬영 위치가 치우친 것이므로, 시료 전체에 걸쳐 고르게 찍어 주세요.")
        else:
            verdict += (f"\n\nℹ️ 사진별 결함률 차이는 입자 수가 적어 생기는 셈 오차 범위 안입니다(p={p_hom:.2f}). "
                        "사진 한 장(판정 입자 수십 개)의 결함률은 ±5%p 안팎 흔들리는 것이 정상이며, 판정은 합산값으로 합니다.")
    verdict += "\n\n" + LEGEND
    table = pd.DataFrame({"항목": list(s.keys()), "값": [fmt(v) for v in s.values()]})
    per_tbl = pd.DataFrame(per_img)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    ax[0].hist(all_df.ecd_um, bins=30, color="#4C78A8"); ax[0].axvline(s["D50 (um)"], color="k", ls="--")
    ax[0].set_xlabel("ECD (um)"); ax[0].set_title(f"Particle size, {n_img} image(s) (D50 {s['D50 (um)']:g} um)")
    cnt = all_df.judge.value_counts()
    keys = ["round", "nonspherical", "defect", "agglomerate", "uncertain"]
    ax[1].bar(["normal\n(round)", "normal\n(elongated)", "defect", "touching", "uncertain"], [cnt.get(k, 0) for k in keys],
              color=["#2E7D32", "#66BB6A", "#C62828", "#4696F0", "#BDBDBD"])
    ax[1].set_title("Particle judgement (" + ("AI" if use_ai else "rule-based") + ")")
    # 사진별 결함률 ± 95% 구간: 한 장의 값이 얼마나 흔들리는지, 합산값(실선)과 기준(점선)을 같이 보여 준다
    xs = np.arange(1, n_img + 1)
    rates = np.array([100 * k / n if n else np.nan for k, n in zip(ks, ns)])
    cis = np.array([ge.wilson(k, n) for k, n in zip(ks, ns)])
    ax[2].errorbar(xs, rates, yerr=np.clip(np.vstack([rates - cis[:, 0], cis[:, 1] - rates]), 0, None),   # 음수 막대 방지
                   fmt="o", color="#C62828", ecolor="#C62828", capsize=3, alpha=0.8)
    ax[2].axhline(s["결함 입자 (%, 판상·균열·조각)"], color="k", lw=1.5, label=f"pooled {s['결함 입자 (%, 판상·균열·조각)']:g}%")
    if n_single:
        ax[2].axhspan(lo, hi, color="k", alpha=0.08)
    ax[2].axhline(defect_max, color="#C62828", ls="--", lw=1, label=f"limit {defect_max:g}%")
    ax[2].set_xticks(xs); ax[2].set_xlabel("image #"); ax[2].set_ylim(0, max(35, float(np.nanmax(cis[:, 1])) + 3))
    ax[2].set_title("Defect % per image (95% CI)"); ax[2].legend(fontsize=8, loc="upper right")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    csv = os.path.join(tempfile.gettempdir(), "NEXTCELL_GraphiteQC_입자측정값.csv")
    all_df.to_csv(csv, index=False, encoding="utf-8-sig")
    return gallery, verdict, table, per_tbl, fig, csv


def run(files, mag, d50_min, d50_max, span_max, circ_min, defect_max, exclude_bad=True, use_circ=True):
    if not files:
        return (None, "SEM 사진을 먼저 올려주세요.", None, None, None, None,
                None, gr.Dropdown(choices=[], value=None), None, "")
    paths = [f if isinstance(f, str) else f.name for f in files]
    mag = int(mag)
    use_ai = (mag == 2000) and classify is not None      # AI 모델은 ×2,000 입자로 학습됨
    entries = analyze_files(paths, mag, use_ai)
    st = dict(entries=entries, mag=mag)
    outs = summarize_entries(entries, mag, d50_min, d50_max, span_max, circ_min, defect_max, exclude_bad, use_circ)
    names = [e["name"] for e in entries]
    first = names[0] if names else None
    img, msg = edit_view(st, first)
    return (*outs, st, gr.Dropdown(choices=names, value=first), img, msg)


# ---------------------------------------------------------------- 결과 고치기 (사람이 클릭해서 판정 바꾸기·되돌리기)
def _entry(st, name):
    if not st or not name:
        return None
    for e in st["entries"]:
        if e["name"] == name:
            return e
    return None


EDIT_MODES = [("정상으로 (초록)", "normal"), ("결함으로 (빨강)", "defect"), ("붙은·겹친 입자로 빼기 (파랑)", "agglomerate"),
              ("판정 불가로 빼기 (회색)", "uncertain"), ("사람이 뺌 (흰 ×)", "manual"), ("자동 판정으로 되돌리기", "auto")]
EDIT_DEFAULT = "사람이 뺌 (흰 ×)"


def _edit_image(e):
    ov = entry_overlay(e)
    f = EDIT_W / ov.shape[1]
    return cv2.resize(ov, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else ov


def _edit_note(e, msg=""):
    df = e["df"]
    n_man = int((df["judge"] == "manual").sum()) if not df.empty else 0
    n_ed = int(df["edited"].astype(bool).sum()) if (not df.empty and "edited" in df) else 0
    n_ov = int(df["overlap"].sum()) if (not df.empty and "overlap" in df) else 0
    head = f"**{e['name']}** · 겹침으로 자동 제외 {n_ov}개 · 사람이 바꾼 입자 {n_ed}개 (그중 뺀 것 {n_man}개)"
    tip = ("  \n위에서 **바꿀 판정을 고른 뒤 입자의 윤곽 안쪽을 클릭**하세요. 초록·빨강·파랑·회색·보라 어느 색이든 바꿀 수 있고, "
           "바꾼 입자에는 흰 점이 찍힙니다. 같은 판정을 고른 채 다시 누르면 자동 판정으로 돌아갑니다. "
           "다 고쳤으면 **고친 내용으로 다시 계산**을 누르세요.")
    return (f"{msg}  \n" if msg else "") + head + tip


def edit_view(st, name):
    e = _entry(st, name)
    if e is None:
        return None, "먼저 사진을 올리고 **검사 시작**을 누르세요."
    return _edit_image(e), _edit_note(e)


def edit_click(st, name, mode, evt: gr.SelectData):
    e = _entry(st, name)
    if e is None or e["df"].empty:
        return None, st, "먼저 사진을 올리고 **검사 시작**을 누르세요."
    try:
        x, y = float(evt.index[0]), float(evt.index[1])
    except Exception:
        return _edit_image(e), st, _edit_note(e, "클릭 위치를 읽지 못했습니다. 다시 눌러 주세요.")
    W = e["parts"]["img"].shape[1]
    f = min(1.0, EDIT_W / (W if W else 1))
    x, y = x / f, y / f                          # 표시 크기 → 분석 해상도 좌표
    if y >= e["parts"]["img"].shape[0]:
        msg = "그림 아래 색 설명 띠입니다. 입자를 눌러 주세요."
    else:
        target = dict(EDIT_MODES).get(mode, "manual")
        msg = "✔ " + ge.set_judge(e["parts"], e["df"], x, y, target)
    return _edit_image(e), st, _edit_note(e, msg)


def edit_reset(st, name):
    e = _entry(st, name)
    if e is None:
        return None, st, "먼저 사진을 올리고 **검사 시작**을 누르세요."
    ge.reset_edits(e["parts"], e["df"])
    return _edit_image(e), st, _edit_note(e, "이 사진에서 고친 것을 모두 취소했습니다.")


def edit_recalc(st, d50_min, d50_max, span_max, circ_min, defect_max, exclude_bad, use_circ):
    if not st:
        return None, "먼저 사진을 올리고 **검사 시작**을 누르세요.", None, None, None, None
    return summarize_entries(st["entries"], st["mag"], d50_min, d50_max, span_max, circ_min, defect_max, exclude_bad, use_circ)


# ---------------------------------------------------------------- 화면

with gr.Blocks(title="흑연 입자 AI 품질검사 · NEXT CELL") as demo:
    gr.Markdown(f"# 흑연 입자 AI 품질검사 · GraphiteQC\nSEM 사진을 올리고 **검사 시작**을 누르세요. 여러 장을 한 번에 올리면 합산해서 판정합니다.  \n"
                f"**{TEAM}** · {SCHOOL} · {MEMBERS}  \n<small>{model_msg} · 버전 {VERSION}</small>")
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="SEM 사진 (여러 장 가능, 하단 정보 바 포함 그대로)", file_count="multiple", file_types=["image"])
            mag = gr.Radio(["2000", "500"], value="2000", label="배율 (×)")
            with gr.Accordion("합격 기준", open=False):
                d50_min = gr.Slider(1, 30, value=6, step=0.5, label="D50 최소 (um)")
                d50_max = gr.Slider(1, 30, value=12, step=0.5, label="D50 최대 (um)")
                span_max = gr.Slider(0.2, 3, value=1.0, step=0.1, label="분포 폭 최대")
                use_circ = gr.Checkbox(value=True, label="평균 원형도 기준 적용 (둥글게 가공한 천연흑연용 · 인조흑연 등 둥글지 않은 제품은 해제)")
                circ_min = gr.Slider(0.5, 1.0, value=0.85, step=0.01, label="평균 원형도 최소 (단독 입자)")
                defect_max = gr.Slider(0, 100, value=15, step=1, label="결함 입자 최대 (%)")
            exclude_bad = gr.Checkbox(value=True, label="테이프 주름 의심 사진은 합산에서 자동 제외")
            btn = gr.Button("검사 시작", variant="primary")
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="입자 판정 결과 (사진별)", columns=2, height=520, object_fit="contain")
            verdict = gr.Markdown()
    with gr.Row():
        table = gr.Dataframe(label="합산 측정 결과")
        plot = gr.Plot(label="분포")
    per_tbl = gr.Dataframe(label="사진별 결과")
    csv = gr.File(label="입자별 측정값 (CSV)")
    st = gr.State(None)
    with gr.Accordion("결과 고치기 — 입자를 클릭해서 판정 바꾸기", open=True):
        edit_mode = gr.Radio([m for m, _ in EDIT_MODES], value=EDIT_DEFAULT, label="입자를 클릭하면 이렇게 바꿉니다")
        with gr.Row():
            edit_pick = gr.Dropdown(choices=[], value=None, label="고칠 사진", interactive=True)
            recalc = gr.Button("고친 내용으로 다시 계산", variant="primary")
            reset = gr.Button("이 사진 고친 것 모두 취소")
        edit_msg = gr.Markdown("먼저 사진을 올리고 **검사 시작**을 누르세요.")
        edit_img = gr.Image(label="입자를 클릭하세요 (같은 판정으로 다시 클릭 = 되돌리기)", type="numpy", interactive=False, height=780)
    criteria = [d50_min, d50_max, span_max, circ_min, defect_max, exclude_bad, use_circ]
    btn.click(run, [files, mag] + criteria, [gallery, verdict, table, per_tbl, plot, csv, st, edit_pick, edit_img, edit_msg])
    edit_pick.change(edit_view, [st, edit_pick], [edit_img, edit_msg])
    edit_img.select(edit_click, [st, edit_pick, edit_mode], [edit_img, st, edit_msg])
    reset.click(edit_reset, [st, edit_pick], [edit_img, st, edit_msg])
    recalc.click(edit_recalc, [st] + criteria, [gallery, verdict, table, per_tbl, plot, csv])
    with gr.Accordion("만든 사람 · 프로그램 정보", open=False):
        gr.Markdown(ABOUT)


def free_port(start=PORT, tries=10):
    """이전 프로그램이 아직 켜져 있으면 그 주소는 쓸 수 없으므로 빈 주소를 찾는다 (새 버전이 확실히 열리도록)"""
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as so:
            try:
                so.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start

if __name__ == "__main__":
    port = free_port()
    print(f"\n흑연 입자 AI 품질검사 · GraphiteQC {VERSION}")
    print(f"만든 사람: {TEAM} ({SCHOOL}) — {MEMBERS}")
    if port != PORT:
        print(f"⚠ 이전에 켠 검사 프로그램이 아직 실행 중입니다. 새 버전은 http://127.0.0.1:{port} 에서 엽니다.")
        print("  (예전 검은 창은 닫아 주세요. 브라우저의 예전 탭도 닫는 것이 좋습니다.)")
    print("검사 화면을 브라우저에서 엽니다...  (이 창을 닫으면 프로그램이 종료됩니다)\n")
    threading.Timer(3, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=False, share=False)
