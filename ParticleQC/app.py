"""
ParticleQC · 범용 입자 AI 품질검사 (NEXT CELL)
- 재료·배율·현미경이 달라도: 스케일과 입자 크기만 알려 주면 입자를 찾고, 사람이 normal / abnormal / exclude 로 나눠 준 입자로
  AI를 학습시켜 시료 단위 품질검사를 한다. 모든 것이 '프로젝트 폴더' 하나에 들어간다.
- 화면: ① 설정·미리보기 → ② 라벨링 준비 → ③ 학습 → ④ 검사
"""
import os, sys, json, socket, tempfile, threading, webbrowser, traceback
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import particle_engine as pe
import particle_tools as pt
import color_legend as cl
from particle_model import ParticleModel
import gradio as gr

PORT = 7861
VERSION = "v1.1.0 (2026-10-30)"
TEAM = "NEXT CELL"
MEMBERS = "윤성빈(팀장) · 김태욱 · 공진우 · 허동현"
SCHOOL = "순천제일대학교 에너지소재화공과"
CREDIT = f"ParticleQC {VERSION} · {TEAM} · {SCHOOL}"
ABOUT = f"""### 만든 사람
**{TEAM}** · {SCHOOL}  
{MEMBERS}

흑연 SEM 사진으로 만든 품질검사 방법(입자 찾기 → 사람이 normal / abnormal / exclude 로 분류 → AI 학습 → 시료 단위 판정)을
다른 입자·배율·현미경에도 쓸 수 있게 일반화한 프로그램입니다. 학습된 모델은 들어 있지 않으며, 쓰는 사람이 자기 시료로 분류해 학습시킵니다.
같은 팀의 흑연 전용 프로그램(GraphiteQC)에서 검증한 방법을 그대로 씁니다 (원래 흑연 결함 비율: AI 11.2% · 사람 10.7%).

버전 {VERSION} · © 2026 {TEAM}. MIT 라이선스로 공개 (프로그램 폴더의 LICENSE 파일). 고쳐 쓰거나 나눠 줄 때는 이 저작권 표시를 남겨 주세요.
"""
DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), "Documents", "ParticleQC", "project1")


def _paths(files):
    return [f if isinstance(f, str) else f.name for f in (files or [])]


def read_rgb(path):
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def fmt(v):
    if isinstance(v, float):
        return "" if v != v else f"{v:g}"
    return str(v)


# ---------------------------------------------------------------- 프로젝트
def open_project(root):
    root = (root or "").strip() or DEFAULT_ROOT
    sp = pt.ensure_project(root)
    s = pe.load_settings(sp)
    st = pt.project_status(root)
    msg = (f"프로젝트: `{root}`\n\n사진 {st['photos']}장 · 분류 전 입자 {st['crops_unsorted']}개 · "
           f"normal {st['normal']} / abnormal {st['abnormal']} / exclude {st['exclude']} · 모델 {'있음' if st['model'] else '없음'}")
    return (root, s, msg, s["um_per_px"], s["d50_um"], s["min_um"], s["max_um"], s["bright_particles"], s["thresh"], s["np_contrast"], s["bar_cut_px"],
            s["abnormal_max"], s["use_size"], s["d50_min"], s["d50_max"], s["span_max"])


def collect_settings(s, um_per_px, d50_um, min_um, max_um, bright, thresh, np_contrast, bar_cut):
    s = dict(s or pe.DEFAULTS)
    s.update(um_per_px=float(um_per_px), d50_um=float(d50_um), min_um=float(min_um), max_um=float(max_um), bright_particles=bool(bright),
             thresh=float(thresh), np_contrast=float(np_contrast), bar_cut_px=int(bar_cut or 0))
    return s


def save_settings(root, s, um_per_px, d50_um, min_um, max_um, bright, thresh, np_contrast, bar_cut, abnormal_max, use_size, d50_min, d50_max, span_max):
    s = collect_settings(s, um_per_px, d50_um, min_um, max_um, bright, thresh, np_contrast, bar_cut)
    s.update(abnormal_max=float(abnormal_max), use_size=bool(use_size), d50_min=float(d50_min), d50_max=float(d50_max), span_max=float(span_max))
    pe.save_settings(os.path.join(root, "settings.json"), s)
    return s, "설정을 저장했습니다."


# ---------------------------------------------------------------- ① 설정·미리보기
def measure_bar(file, bar_um):
    if not file:
        return "먼저 미리보기 사진을 올려 주세요.", None
    g = pe.read_gray(_paths([file])[0])
    px = pe.detect_scale_bar(g)
    if not px:
        return "사진 아래쪽에서 스케일 바를 찾지 못했습니다. 다른 프로그램(ImageJ 등)으로 바 길이를 재서 픽셀 수를 직접 넣어 주세요.", None
    if bar_um and float(bar_um) > 0:
        return f"스케일 바 길이 {px} px → {float(bar_um) / px:.5f} µm/px 로 설정했습니다. 아래 '설정 저장'을 눌러 주세요.", float(bar_um) / px
    return f"스케일 바 길이 {px} px. 바가 몇 µm인지 입력하고 다시 누르세요.", None


def preview(file, s, um_per_px, d50_um, min_um, max_um, bright, thresh, np_contrast, bar_cut):
    if not file:
        return None, "미리보기 사진을 올려 주세요."
    s = collect_settings(s, um_per_px, d50_um, min_um, max_um, bright, thresh, np_contrast, bar_cut)
    p = _paths([file])[0]
    try:
        ov, df = pe.analyze(p, s, None)
        ov = cl.add_legend(ov, pe.LEGEND_PREVIEW, credit=CREDIT)          # 그림 아래에 색 설명
    except Exception as e:
        return None, f"분석 오류: {e}"
    if df.empty:
        return ov, "입자를 찾지 못했습니다. 스케일(µm/px)·입자 크기·밝기 문턱을 확인해 주세요. 입자가 배경보다 어두우면 '입자가 밝음' 체크를 해제하세요."
    d50 = float(np.median(df.ecd_um)); n_single = int((df.judge == "single").sum())   # 8자 모양(겹친 두 입자)은 붙은 입자로 셈
    msg = (f"찾은 입자 {len(df)}개 (단독 {n_single}개, 붙은 입자 {len(df) - n_single}개) · 입자 아님 {df.attrs.get('n_not_particle', 0)}개\n\n"
           f"측정 D50 = {d50:.2f} µm (설정한 예상 크기 {float(d50_um):g} µm)")
    if abs(d50 - float(d50_um)) > 0.3 * float(d50_um):
        msg += f"\n\n⚠️ 측정된 크기가 예상 크기와 많이 다릅니다. '예상 입자 크기'를 {d50:.1f} 근처로 바꾸고 다시 미리보기 하세요 (스케일이 틀렸을 수도 있습니다)."
    if n_single < 20:
        msg += "\n\nℹ️ 단독 입자가 적습니다. 시료를 더 얇게 펴거나 입자 수가 많은 자리를 찍으면 한 장당 판정 수가 늘어납니다."
    msg += "\n\n초록 = 단독 입자(판정 대상) · 파랑 = 붙은 입자(제외) · 보라 얇은 선 = 테이프 주름(입자 아님, 제외). 윤곽이 입자에 잘 맞으면 설정을 저장하세요."
    return ov, msg


# ---------------------------------------------------------------- ② 라벨링 준비
def do_ingest(root, s, files, sample):
    paths = _paths(files)
    if not paths:
        return "사진을 올려 주세요.", []
    logs = []
    try:
        pt.ensure_project(root)                      # 프로젝트 하위 폴더가 지워졌어도 다시 만든다
        pt.ingest(root, paths, s, sample=(sample or "A").strip() or "A", log=logs.append)
    except Exception as e:
        return "오류: " + str(e) + "\n" + traceback.format_exc(), []
    lab = os.path.join(root, "labeling")
    sheets = sorted(os.path.join(lab, "sheets", f) for f in os.listdir(os.path.join(lab, "sheets")) if f.startswith("sheet_"))
    msg = "\n".join(logs) + (f"\n\n이제 탐색기에서 `{os.path.join(lab, '_crops')}` 의 그림을 보면서 "
                             f"`normal` / `abnormal` / `exclude` 폴더로 옮겨 주세요 (시트의 번호 = 그림 순서). "
                             "exclude = 겹침·흐림·입자 아님·판단 불가. 다 옮기면 ③ 학습으로.")
    return msg, sheets


# ---------------------------------------------------------------- ③ 학습
def do_train(root):
    logs = []
    try:
        rep = pt.train(root, log=logs.append)
    except Exception as e:
        return "\n".join(logs) + "\n\n오류: " + str(e)
    ab = rep["abnormal"]; ex = rep["exclude"]
    out = "\n".join(logs) + "\n\n### 학습 결과 (사진 단위 교차검증)\n"
    out += f"- 라벨: normal {rep['per_class']['normal']} · abnormal {rep['per_class']['abnormal']} · exclude {rep['per_class']['exclude']} · 사진 {rep['n_images']}장\n"
    out += f"- abnormal 입자 하나하나: 정밀도 {ab['precision']:.2f} · 재현율 {ab['recall']:.2f} (실제 {ab['n_true']}개, 예측 {ab['n_pred']}개)\n"
    if ex["caught"] is not None:
        out += f"- 판정 불가(exclude) 걸러냄 {100 * ex['caught']:.0f}% · 정상/결함 라벨이 판정 불가로 빠진 비율 {100 * ex['labeled_lost']:.0f}%\n"
    if "per_sample" in rep:
        out += "- 시료별 abnormal 비율 (AI 예측 vs 사람 라벨): " + " · ".join(
            f"{k}: {v['abnormal_model']:.1f}% vs {v['abnormal_label']:.1f}%" for k, v in rep["per_sample"].items() if v["abnormal_model"] is not None) + "\n"
    out += f"- 기준값: 판정불가 {rep['thr_exclude']:.2f} · abnormal {rep['thr_defect']:.2f}\n\n모델이 `model.npz` 로 저장됐고 ④ 검사에서 바로 씁니다."
    return out


# ---------------------------------------------------------------- ④ 검사
EDIT_W = 1600          # '결과 고치기' 그림의 표시 가로 크기(px). 클릭 좌표는 이 비율로 되돌려 쓴다


def inspect_files(paths, s, classify):
    """사진마다 분석해 (이름, 입자 표, 그림 부품)을 모은다. 사람이 고친 뒤 다시 그리고 다시 계산할 때 이것을 쓴다"""
    entries, seen = [], {}
    for p in paths:
        name = os.path.basename(p)
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        try:
            _, df, parts = pe.analyze(p, s, classify, return_parts=True)
        except Exception as e:
            entries.append(dict(name=name, df=pd.DataFrame(), parts=None, error=str(e)))
            continue
        if not df.empty:
            df.insert(0, "image", name)
        entries.append(dict(name=name, df=df, parts=parts, error=None))
    return entries


def entry_overlay(e, has_model):
    if e["parts"] is None:
        return np.zeros((10, 10, 3), np.uint8)
    legend = (pe.LEGEND_INSPECT if has_model else pe.LEGEND_PREVIEW) + [pe.LEGEND_MANUAL]
    return cl.add_legend(pe.render(e["parts"], e["df"]), legend, credit=CREDIT)


def summarize_entries(entries, s, has_model, abnormal_max, exclude_bad):
    plt.close("all")
    gallery, dfs, per = [], [], []
    for e in entries:
        name, df = e["name"], e["df"]
        if e["error"]:
            gallery.append((entry_overlay(e, has_model), f"{name}: 오류 {e['error']}")); continue
        ov = entry_overlay(e, has_model)
        if df.empty:
            gallery.append((ov, f"{name}: 입자 없음")); continue
        dfs.append(df)
        st, _, _ = pe.summarize(df, s)
        lo, hi = pe.wilson(st["abnormal 입자 수"], st["판정 대상 (단독 입자)"])
        per.append({"사진": name, "품질": "", "입자 수": st["입자 수"], "입자 아님": st["입자 아님 (제외)"], "D50 (um)": st["D50 (um)"],
                    "판정 입자": st["판정 대상 (단독 입자)"], "abnormal (개)": st["abnormal 입자 수"], "abnormal (%)": st["abnormal 입자 (%)"],
                    "95% 구간": f"{lo:.0f}~{hi:.0f}%" if st["판정 대상 (단독 입자)"] else "",
                    "접촉 (%)": st["응집·접촉 입자 (%, 판정 제외)"], "판정불가 (%)": st["판정 불가 (%, 겹침·흐림 등)"],
                    "겹침 자동 제외": st["겹친 입자로 뺀 수 (자동)"], "사람이 바꿈": st["사람이 판정을 바꾼 수"],
                    "사람이 뺌": st["사람이 직접 뺀 수"]})
        gallery.append((ov, f"{name} · abnormal {st['abnormal 입자 (%)']:g}% · D50 {st['D50 (um)']:g} um"))
    if not dfs:
        return gallery, "입자를 찾지 못했습니다. ① 설정·미리보기에서 스케일과 입자 크기를 확인해 주세요.", None, None, None, None
    # 사진 품질 선별 (입자 수가 다른 사진의 절반 이하 / 입자 아님 30% 이상 → 주름·오염 의심)
    med = float(np.median([r["입자 수"] for r in per])); bad = []
    for i, r in enumerate(per):
        cand = r["입자 수"] + r["입자 아님"]
        if (len(per) >= 3 and r["입자 수"] < 0.5 * med) or (cand and r["입자 아님"] / cand >= 0.30):
            bad.append(i)
    note = ""
    if bad:
        names = ", ".join(per[i]["사진"] for i in bad)
        if exclude_bad and len(dfs) - len(bad) >= 3:
            for i in bad:
                per[i]["품질"] = "⚠ 의심 → 제외"
            dfs = [d for i, d in enumerate(dfs) if i not in bad]; used = [r for i, r in enumerate(per) if i not in bad]
            note = f"\n\n⚠️ 입자를 제대로 못 찾은 사진 {len(bad)}장(주름·오염·초점 의심)을 합산에서 제외했습니다: {names}"
        else:
            for i in bad:
                per[i]["품질"] = "⚠ 의심"
            used = per; note = f"\n\n⚠️ 입자를 제대로 못 찾은 것으로 의심되는 사진: {names}"
    else:
        used = per
    all_df = pd.concat(dfs, ignore_index=True); all_df.attrs["n_not_particle"] = sum(d.attrs.get("n_not_particle", 0) for d in dfs)
    st, checks, ok = pe.summarize(all_df, s)
    n_img = len(dfs)
    ks = [r["abnormal (개)"] for r in used]; ns = [r["판정 입자"] for r in used]
    n_single, k_ab = int(sum(ns)), int(sum(ks))
    lo, hi = pe.wilson(k_ab, n_single); half = (hi - lo) / 2 if n_single else float("nan")
    disp, p_hom = pe.homogeneity(ks, ns)
    method = "AI (normal / abnormal / 판정불가)" if has_model else "AI 모델 없음 — ③ 학습을 먼저 하세요 (지금은 입자 찾기·크기만 표시)"
    borderline = bool(n_single) and lo <= float(abnormal_max) <= hi and all(v for k, v in checks.items() if "abnormal" not in k)
    head = "⚠️ 판정 보류 (사진 추가 필요)" if borderline else ("✅ 합격" if ok else "❌ 불합격")
    if not has_model:
        head = "ℹ️ 모델 없음 (판정 안 함)"
    verdict = f"## {head}\n사진 {n_img}장 합산 · 판정 방법: **{method}**\n\n" + "\n".join(f"- {'✅' if v else '❌'} {k}" for k, v in checks.items()) + note
    n_man = int((all_df["judge"] == "manual").sum())
    n_ed = int(all_df["edited"].astype(bool).sum()) if "edited" in all_df else 0
    if n_ed:
        verdict += f"\n\n✋ 사람이 판정을 바꾼 입자 {n_ed}개(그중 뺀 것 {n_man}개)를 반영한 결과입니다."
    if n_single and has_model:
        verdict += f"\n\n**abnormal 비율 {st['abnormal 입자 (%)']:g}%** (abnormal {k_ab}개 / 판정 입자 {n_single}개 · 95% 신뢰구간 {lo:.1f}~{hi:.1f}%, ±{half:.1f}%p)"
        p_ = max(k_ab, 1) / n_single; need = int(np.ceil(p_ * (1 - p_) / (0.03 / 1.96) ** 2 / max(n_single / n_img, 1)))
        if lo <= float(abnormal_max) <= hi:
            verdict += f"\n\n⚠️ 신뢰구간이 기준 {float(abnormal_max):g}%에 걸쳐 있어 사진을 더 찍으면 판정이 바뀔 수 있습니다. 같은 시료를 {max(need - n_img, 3)}장 이상 더 찍어 함께 올리세요."
        elif n_img < 10:
            verdict += f"\n\n⚠️ 사진 {n_img}장(판정 입자 {n_single}개)은 abnormal 비율이 ±{half:.1f}%p 흔들립니다. ±3%p 안으로 줄이려면 약 {need}장을 찍어 함께 올리세요."
        if p_hom == p_hom and n_img >= 5:
            verdict += (f"\n\nℹ️ 사진별 abnormal 비율 차이가 입자 수로 설명되는 범위보다 큽니다(산포 {disp:.1f}배, p={p_hom:.3f}). 시료가 고르지 않거나 촬영 위치가 치우친 것입니다."
                        if p_hom < 0.05 else f"\n\nℹ️ 사진별 차이는 셈 오차 범위 안입니다(p={p_hom:.2f}). 판정은 합산값으로 합니다.")
    verdict += "\n\n초록 = normal · 빨강 = abnormal · 파랑 = 붙은·겹친 입자(제외) · 회색 = 판정 불가(제외) · 보라 얇은 선 = 테이프 주름(입자 아님, 제외) · 흰색 × = 사람이 뺀 입자 · 흰 점 = 사람이 판정을 바꾼 입자"
    table = pd.DataFrame({"항목": list(st.keys()), "값": [fmt(v) for v in st.values()]})
    per_tbl = pd.DataFrame(per)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    ax[0].hist(all_df.ecd_um, bins=30, color="#4C78A8"); ax[0].axvline(st["D50 (um)"], color="k", ls="--")
    ax[0].set_xlabel("ECD (um)"); ax[0].set_title(f"Particle size, {n_img} image(s) (D50 {st['D50 (um)']:g} um)")
    cnt = all_df.judge.value_counts()
    keys = ["normal", "abnormal", "agglomerate", "uncertain", "single"]
    ax[1].bar(["normal", "abnormal", "touching", "uncertain", "unjudged"], [cnt.get(k, 0) for k in keys], color=["#2E7D32", "#C62828", "#4696F0", "#BDBDBD", "#888888"])
    ax[1].set_title("Particle judgement")
    xs = np.arange(1, n_img + 1); rates = np.array([100 * k / n if n else np.nan for k, n in zip(ks, ns)])
    cis = np.array([pe.wilson(k, n) for k, n in zip(ks, ns)]) if ns else np.zeros((0, 2))
    if len(cis) and np.isfinite(cis).any():
        ax[2].errorbar(xs[:len(rates)], rates, yerr=np.clip(np.vstack([rates - cis[:, 0], cis[:, 1] - rates]), 0, None),   # 음수 막대 방지
                       fmt="o", color="#C62828", capsize=3)
        ax[2].set_ylim(0, max(35, float(np.nanmax(cis[:, 1])) + 3))
    ax[2].axhline(st["abnormal 입자 (%)"], color="k", lw=1.5, label=f"pooled {st['abnormal 입자 (%)']:g}%")
    ax[2].axhline(float(abnormal_max), color="#C62828", ls="--", lw=1, label=f"limit {float(abnormal_max):g}%")
    ax[2].set_xticks(xs); ax[2].set_xlabel("image #"); ax[2].set_title("abnormal % per image (95% CI)"); ax[2].legend(fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    csv = os.path.join(tempfile.gettempdir(), "NEXTCELL_ParticleQC_입자측정값.csv")
    all_df.to_csv(csv, index=False, encoding="utf-8-sig")
    return gallery, verdict, table, per_tbl, fig, csv




def inspect(root, s, files, abnormal_max, use_size, d50_min, d50_max, span_max, exclude_bad):
    paths = _paths(files)
    if not paths:
        return (None, "사진을 올려 주세요.", None, None, None, None, None, gr.Dropdown(choices=[], value=None), None, "")
    s = dict(s); s.update(abnormal_max=float(abnormal_max), use_size=bool(use_size), d50_min=float(d50_min), d50_max=float(d50_max), span_max=float(span_max))
    mp = os.path.join(root, "model.npz")
    classify = ParticleModel(mp).predict if os.path.exists(mp) else None
    entries = inspect_files(paths, s, classify)
    qs = dict(entries=entries, s=s, has_model=classify is not None)
    outs = summarize_entries(entries, s, qs["has_model"], abnormal_max, exclude_bad)
    names = [e["name"] for e in entries if e["parts"] is not None]
    first = names[0] if names else None
    img, msg = edit_view(qs, first)
    return (*outs, qs, gr.Dropdown(choices=names, value=first), img, msg)


# ---------------------------------------------------------------- 결과 고치기 (사람이 클릭해서 판정 바꾸기·되돌리기)
def _entry(qs, name):
    if not qs or not name:
        return None
    for e in qs["entries"]:
        if e["name"] == name and e["parts"] is not None:
            return e
    return None


EDIT_MODES = [("normal 로 (초록)", "normal"), ("abnormal 로 (빨강)", "abnormal"), ("붙은·겹친 입자로 빼기 (파랑)", "agglomerate"),
              ("판정 불가로 빼기 (회색)", "uncertain"), ("사람이 뺌 (흰 ×)", "manual"), ("자동 판정으로 되돌리기", "auto")]
EDIT_DEFAULT = "사람이 뺌 (흰 ×)"


def _edit_image(qs, e):
    ov = entry_overlay(e, qs["has_model"])
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


def edit_view(qs, name):
    e = _entry(qs, name)
    if e is None:
        return None, "먼저 사진을 올리고 **검사 시작**을 누르세요."
    return _edit_image(qs, e), _edit_note(e)


def edit_click(qs, name, mode, evt: gr.SelectData):
    e = _entry(qs, name)
    if e is None or e["df"].empty:
        return None, qs, "먼저 사진을 올리고 **검사 시작**을 누르세요."
    try:
        x, y = float(evt.index[0]), float(evt.index[1])
    except Exception:
        return _edit_image(qs, e), qs, _edit_note(e, "클릭 위치를 읽지 못했습니다. 다시 눌러 주세요.")
    W, H = e["parts"]["img"].shape[1], e["parts"]["img"].shape[0]
    f = min(1.0, 2600.0 / W) * min(1.0, EDIT_W / min(W, 2600.0))     # 분석 해상도 → 표시 크기 비율
    x, y = x / f, y / f
    if y >= H:
        msg = "그림 아래 색 설명 띠입니다. 입자를 눌러 주세요."
    else:
        target = dict(EDIT_MODES).get(mode, "manual")
        msg = "✔ " + pe.set_judge(e["parts"], e["df"], x, y, target, has_model=qs["has_model"])
    return _edit_image(qs, e), qs, _edit_note(e, msg)


def edit_reset(qs, name):
    e = _entry(qs, name)
    if e is None:
        return None, qs, "먼저 사진을 올리고 **검사 시작**을 누르세요."
    pe.reset_edits(e["parts"], e["df"])
    return _edit_image(qs, e), qs, _edit_note(e, "이 사진에서 고친 것을 모두 취소했습니다.")


def edit_recalc(qs, s, abnormal_max, use_size, d50_min, d50_max, span_max, exclude_bad):
    if not qs:
        return None, "먼저 사진을 올리고 **검사 시작**을 누르세요.", None, None, None, None
    s = dict(qs["s"]); s.update(abnormal_max=float(abnormal_max), use_size=bool(use_size), d50_min=float(d50_min), d50_max=float(d50_max), span_max=float(span_max))
    return summarize_entries(qs["entries"], s, qs["has_model"], abnormal_max, exclude_bad)


# ---------------------------------------------------------------- 화면

with gr.Blocks(title="ParticleQC · 범용 입자 품질검사 · NEXT CELL") as demo:
    gr.Markdown("# ParticleQC · 범용 입자 AI 품질검사\n재료·배율·현미경이 달라도, 스케일과 입자 크기를 알려 주고 입자를 normal / abnormal / exclude 로 나눠 주면 검사할 수 있습니다.  \n"
                f"**{TEAM}** · {SCHOOL} · {MEMBERS}  \n<small>버전 {VERSION}</small>")
    settings = gr.State(dict(pe.DEFAULTS))
    with gr.Row():
        root = gr.Textbox(value=DEFAULT_ROOT, label="프로젝트 폴더 (재료마다 하나. 없으면 만들어집니다)", scale=4)
        open_btn = gr.Button("프로젝트 열기 / 만들기", variant="primary", scale=1)
    status = gr.Markdown("프로젝트 폴더를 정하고 '프로젝트 열기'를 누르세요.")

    with gr.Tabs():
        with gr.Tab("① 설정 · 미리보기"):
            gr.Markdown("사진 한 장으로 **스케일**과 **입자 크기**를 맞추고, 입자를 제대로 찾는지 확인한 뒤 저장합니다.")
            with gr.Row():
                with gr.Column(scale=1):
                    pv_file = gr.File(label="미리보기 사진 1장", file_count="single", file_types=["image"])
                    with gr.Group():
                        gr.Markdown("**스케일** — 사진 아래 스케일 바 길이(µm)를 넣고 '자동 측정', 안 되면 µm/px 직접 입력")
                        with gr.Row():
                            bar_um = gr.Number(value=30, label="스케일 바 (µm)")
                            bar_btn = gr.Button("스케일 바 자동 측정")
                        um_per_px = gr.Number(value=pe.DEFAULTS["um_per_px"], label="µm / 픽셀 (원본 사진 기준)", precision=5)
                        bar_msg = gr.Markdown("")
                    with gr.Group():
                        d50_um = gr.Number(value=pe.DEFAULTS["d50_um"], label="예상 입자 크기 D50 (µm)")
                        with gr.Row():
                            min_um = gr.Number(value=pe.DEFAULTS["min_um"], label="최소 (µm)")
                            max_um = gr.Number(value=pe.DEFAULTS["max_um"], label="최대 (µm)")
                        bright = gr.Checkbox(value=True, label="입자가 배경보다 밝음 (SEM SE). 광학현미경처럼 어두우면 해제")
                        thresh = gr.Slider(3, 60, value=pe.DEFAULTS["thresh"], step=1, label="밝기 문턱 (입자를 놓치면 낮추고, 배경을 잡으면 높임)")
                        np_contrast = gr.Slider(0, 80, value=pe.DEFAULTS["np_contrast"], step=1, label="'입자 아님' 밝기 기준 (이보다 어둡고 윤곽도 흐릿하면 테이프 주름으로 제외)")
                        bar_cut = gr.Number(value=0, label="아래 정보 바 높이 (px, 0 = 자동)")
                    pv_btn = gr.Button("미리보기", variant="primary")
                with gr.Column(scale=2):
                    pv_img = gr.Image(label="입자 찾기 결과", type="numpy")
                    pv_msg = gr.Markdown("")
            gr.Markdown("---\n**합격 기준** (④ 검사에서 씀)")
            with gr.Row():
                abnormal_max = gr.Number(value=pe.DEFAULTS["abnormal_max"], label="abnormal 입자 최대 (%)")
                use_size = gr.Checkbox(value=False, label="크기 기준도 적용")
                d50_min = gr.Number(value=pe.DEFAULTS["d50_min"], label="D50 최소 (µm)")
                d50_max = gr.Number(value=pe.DEFAULTS["d50_max"], label="D50 최대 (µm)")
                span_max = gr.Number(value=pe.DEFAULTS["span_max"], label="분포 폭 최대")
            save_btn = gr.Button("설정 저장", variant="primary")
            save_msg = gr.Markdown("")

        with gr.Tab("② 라벨링 준비"):
            gr.Markdown("사진을 올리면 단독 입자를 하나씩 잘라 `labeling/_crops` 에 넣고, 한눈에 보는 시트를 만듭니다. "
                        "그 다음 탐색기에서 그림을 `normal` / `abnormal` / `exclude` 폴더로 옮기세요. "
                        "**권장: abnormal 100개 이상, 전체 1,000개 이상.** exclude(겹침·흐림·입자 아님)도 꼭 넣으세요.")
            with gr.Row():
                ing_files = gr.File(label="사진 (여러 장)", file_count="multiple", file_types=["image"])
                sample = gr.Textbox(value="A", label="시료 이름 (예: 정상, 손상1 — 학습 결과에 시료별 비율이 나옵니다)")
            ing_btn = gr.Button("입자 잘라내기", variant="primary")
            ing_msg = gr.Markdown("")
            sheets = gr.Gallery(label="라벨링 시트 (번호 = 그림 순서)", columns=2, height=480, object_fit="contain")

        with gr.Tab("③ 학습"):
            gr.Markdown("세 폴더에 그림을 다 옮겼으면 누르세요. 5~10분 걸립니다 (입자 1,000개 기준). 끝나면 `model.npz` 가 생기고 ④ 검사에서 바로 씁니다.")
            tr_btn = gr.Button("학습 시작", variant="primary")
            tr_msg = gr.Markdown("")

        with gr.Tab("④ 검사"):
            gr.Markdown("같은 조건으로 찍은 사진을 여러 장(10장 이상 권장) 올리면 합산해서 판정합니다. 합격 기준은 ① 탭에서 저장한 값입니다.")
            with gr.Row():
                with gr.Column(scale=1):
                    in_files = gr.File(label="검사할 사진 (여러 장)", file_count="multiple", file_types=["image"])
                    exclude_bad = gr.Checkbox(value=True, label="입자를 제대로 못 찾은 사진(주름·오염 의심)은 자동 제외")
                    in_btn = gr.Button("검사 시작", variant="primary")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="입자 판정 결과 (사진별)", columns=2, height=520, object_fit="contain")
                    verdict = gr.Markdown()
            with gr.Row():
                table = gr.Dataframe(label="합산 측정 결과")
                plot = gr.Plot(label="분포")
            per_tbl = gr.Dataframe(label="사진별 결과")
            csv = gr.File(label="입자별 측정값 (CSV)")
            qstate = gr.State(None)
            with gr.Accordion("결과 고치기 — 입자를 클릭해서 판정 바꾸기", open=True):
                edit_mode = gr.Radio([m for m, _ in EDIT_MODES], value=EDIT_DEFAULT, label="입자를 클릭하면 이렇게 바꿉니다")
                with gr.Row():
                    edit_pick = gr.Dropdown(choices=[], value=None, label="고칠 사진", interactive=True)
                    recalc = gr.Button("고친 내용으로 다시 계산", variant="primary")
                    reset = gr.Button("이 사진 고친 것 모두 취소")
                edit_msg = gr.Markdown("먼저 사진을 올리고 **검사 시작**을 누르세요.")
                edit_img = gr.Image(label="입자를 클릭하세요 (같은 판정으로 다시 클릭 = 되돌리기)", type="numpy", interactive=False, height=780)

    open_btn.click(open_project, [root], [root, settings, status, um_per_px, d50_um, min_um, max_um, bright, thresh, np_contrast, bar_cut,
                                         abnormal_max, use_size, d50_min, d50_max, span_max])
    bar_btn.click(measure_bar, [pv_file, bar_um], [bar_msg, um_per_px])
    pv_btn.click(preview, [pv_file, settings, um_per_px, d50_um, min_um, max_um, bright, thresh, np_contrast, bar_cut], [pv_img, pv_msg])
    save_btn.click(save_settings, [root, settings, um_per_px, d50_um, min_um, max_um, bright, thresh, np_contrast, bar_cut,
                                   abnormal_max, use_size, d50_min, d50_max, span_max], [settings, save_msg])
    ing_btn.click(do_ingest, [root, settings, ing_files, sample], [ing_msg, sheets])
    tr_btn.click(do_train, [root], [tr_msg])
    in_btn.click(inspect, [root, settings, in_files, abnormal_max, use_size, d50_min, d50_max, span_max, exclude_bad],
                 [gallery, verdict, table, per_tbl, plot, csv, qstate, edit_pick, edit_img, edit_msg])
    edit_pick.change(edit_view, [qstate, edit_pick], [edit_img, edit_msg])
    edit_img.select(edit_click, [qstate, edit_pick, edit_mode], [edit_img, qstate, edit_msg])
    reset.click(edit_reset, [qstate, edit_pick], [edit_img, qstate, edit_msg])
    recalc.click(edit_recalc, [qstate, settings, abnormal_max, use_size, d50_min, d50_max, span_max, exclude_bad],
                 [gallery, verdict, table, per_tbl, plot, csv])
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
    print(f"\nParticleQC {VERSION} · 범용 입자 AI 품질검사")
    print(f"만든 사람: {TEAM} ({SCHOOL}) — {MEMBERS}")
    if port != PORT:
        print(f"⚠ 주소 {PORT} 를 다른 프로그램(예전 ParticleQC 또는 흑연 검사)이 쓰고 있어 http://127.0.0.1:{port} 에서 엽니다.")
    print("ParticleQC 화면을 브라우저에서 엽니다...  (이 창을 닫으면 프로그램이 종료됩니다)\n")
    threading.Timer(3, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=False, share=False)
