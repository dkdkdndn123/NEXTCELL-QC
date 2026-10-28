@echo off
cd /d "%~dp0"
title 흑연 품질검사 - 처음 설치
echo.
echo  ==========================================
echo   흑연 입자 AI 품질검사 (NEXT CELL) 설치
echo   처음 한 번만 하면 됩니다. 5~10분 걸릴 수 있어요.
echo  ==========================================
echo.

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  python -c "import sys" >nul 2>nul && set "PY=python"
)
if not defined PY goto nopython

%PY% -c "import sys; print('Python', sys.version.split()[0])"
%PY% -c "import sys; sys.exit(0 if (3,9)<=sys.version_info[:2]<=(3,13) else 1)"
if errorlevel 1 (
  echo.
  echo  [안내] Python 3.9~3.13 버전이 필요합니다. 3.12 설치를 권장합니다.
  echo  https://www.python.org/downloads/release/python-3128/
  echo  설치할 때 "Add python.exe to PATH" 를 꼭 체크하세요.
  pause
  exit /b 1
)

echo.
echo  [1/3] 전용 실행 환경 만드는 중...
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
if errorlevel 1 goto fail

echo  [2/3] 기본 패키지 설치 중...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo  [3/3] 바탕화면 아이콘 만드는 중...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_shortcut.ps1"

echo.
echo  설치 완료! 바탕화면의 "흑연 품질검사" 아이콘을 더블클릭하세요.
echo.
pause
exit /b 0

:nopython
echo  [안내] Python이 설치되어 있지 않습니다.
echo  아래 주소에서 Python 3.12를 설치한 뒤 이 파일을 다시 실행하세요.
echo  https://www.python.org/downloads/release/python-3128/
echo  설치할 때 "Add python.exe to PATH" 를 꼭 체크하세요.
pause
exit /b 1

:fail
echo.
echo  [오류] 설치 중 문제가 생겼습니다. 인터넷 연결을 확인하고 다시 실행해 주세요.
echo  계속 안 되면 이 창의 내용을 캡처해서 보내주세요.
pause
exit /b 1
