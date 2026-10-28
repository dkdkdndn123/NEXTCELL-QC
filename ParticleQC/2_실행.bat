@echo off
cd /d "%~dp0"
title ParticleQC 입자 품질검사 (이 창을 닫으면 종료됩니다)
if not exist ".venv\Scripts\python.exe" (
  echo 먼저 "1_처음_설치.bat" 을 실행해 주세요.
  pause
  exit /b 1
)
echo 프로그램을 시작하는 중입니다. 잠시 후 브라우저가 열립니다...
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
