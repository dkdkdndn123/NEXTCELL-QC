"""
새 ×2,000 사진 → 단독 입자 잘라내기 → 라벨링 시트 만들기
검사 화면의 'AI 다시 학습' → '입자 잘라 내기' 버튼과 같은 일을 명령으로 합니다 (계산은 graphite_tools.ingest).
사용법: python ingest_new_photos.py <사진 폴더> <출력 폴더> <접두어>
  - 출력: <출력 폴더>/crops/<접두어>_<사진>_pNNNN.png, crops.csv, 시트 sheet_XX.png
  - 접촉 필터를 통과한 단독 입자만 자름 (응집·접촉 입자는 제외)
  - 이미 있는 출력 폴더에 다시 하면 이어 붙임 (같은 접두어의 같은 사진은 건너뜀). 접두어는 crops.csv 의 sample 열에도 들어감
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graphite_tools as gt

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    src, out, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    photos = sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.tif")))
    gt.ingest(out, photos, sample=prefix, log=lambda s: print(s, flush=True))
