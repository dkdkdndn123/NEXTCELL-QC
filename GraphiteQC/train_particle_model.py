"""
입자 4분류 모델 학습: round / nonspherical / defect / exclude(판정 불가)
검사 화면의 'AI 다시 학습' → '학습 시작' 버튼과 같은 일을 명령으로 합니다 (계산은 graphite_tools.train).
사용법: python train_particle_model.py <labeling 폴더> <출력 npz> [--cache <특징 캐시 npz>]
- labeling 폴더 구조: round/, nonspherical/, defect/, exclude/ + crops.csv
  (정상/결함/제외 3분류로 쓰려면 정상을 round/ 에 넣고 nonspherical/ 은 비워 두면 됨 — 비구형 판정이 꺼진 모델이 만들어짐)
  crops.csv 필수 열: file, image, ecd_um, circularity, aspect_ratio, solidity  (선택: sample)
- 사진 단위 5겹 교차검증으로 성능을 낸 뒤, 전체 데이터로 최종 모델 학습
- 기준값: 판정불가(thr_exclude)와 결함(thr_defect)은 결함 F1 최대(±0.01) 범위에서 예측 결함 수가 실제와 가장 가까운 값, 비구형(thr_nonspherical)은 매튜스 상관계수(MCC) 최대 — 구형이 훨씬 많아도 비구형 비율이 라벨과 맞게 잡힘
- sample 열이 있으면 시료별(원본/손상 등) 결함 비율을 교차검증 예측과 라벨로 비교해 출력
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graphite_tools as gt

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    LAB, OUT = sys.argv[1], sys.argv[2]
    CACHE = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else None
    rep = gt.train(LAB, OUT, cache=CACHE, log=lambda s: print(s, flush=True))
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    print("saved", OUT)
