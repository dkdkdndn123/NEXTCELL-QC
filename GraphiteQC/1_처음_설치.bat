@echo off
cd /d "%~dp0"
title 흑연 품질검사 - 처음 설치
echo.
echo  ==========================================
echo   흑연 입자 AI 품질검사 (NEXT CELL) 설치
echo   처음 한 번만 하면 됩니다. 5~10분 걸릴 수 있어요.
echo  ==========================================
echo.

rem  이 프로그램은 Python 3.12 에 맞춰 만들었고 3.9~3.13 에서 동작합니다.
rem  다른 버전(3.14, 3.15 등)이 깔려 있어도 3.12 를 먼저 찾아 쓰고, 없으면 3.12 를 자동으로 설치합니다.
set "MGRTRIED="
set "DLTRIED="

:findpy
set "PY="
call :trypy 3.12
call :trypy 3.13
call :trypy 3.11
call :trypy 3.10
call :trypy 3.9
call :trypath "%LocalAppData%\Programs\Python\Python312\python.exe"
call :trypath "%ProgramFiles%\Python312\python.exe"
call :trypath "%LocalAppData%\Python\pythoncore-3.12-64\python.exe"
call :trypath "%LocalAppData%\Programs\Python\Python313\python.exe"
call :trypath "%ProgramFiles%\Python313\python.exe"
call :trypath "%LocalAppData%\Python\pythoncore-3.13-64\python.exe"
call :trypath "%LocalAppData%\Programs\Python\Python311\python.exe"
call :trypath "%ProgramFiles%\Python311\python.exe"
call :trypath "%LocalAppData%\Programs\Python\Python310\python.exe"
call :trypath "%ProgramFiles%\Python310\python.exe"
call :trycmd
call :trymgr
if not defined PY goto nopython

echo  사용할 Python:
%PY% -c "import sys; print('  Python', sys.version.split()[0])"

echo.
echo  [1/3] 전용 실행 환경 만드는 중...
if not exist ".venv\Scripts\python.exe" goto makevenv
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,9)<=sys.version_info[:2]<=(3,13) else 1)" <nul >nul 2>nul
if not errorlevel 1 goto pipinstall
echo  전에 만든 실행 환경의 Python 버전이 맞지 않아 새로 만듭니다.
rmdir /s /q ".venv"
:makevenv
%PY% -m venv .venv
if errorlevel 1 goto fail

:pipinstall
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
echo.
echo  [안내] 이 프로그램은 Python 3.12 에 맞춰 만들었습니다. (3.9~3.13 도 됩니다)
echo  이 컴퓨터에서는 맞는 버전을 찾지 못했습니다. 3.14 이상은 아직 시험하지 않아 쓰지 않습니다.
echo.
if defined DLTRIED goto manual
if defined MGRTRIED goto dlpython
echo  지금 있는 Python 은 그대로 두고, 이 프로그램이 쓸 Python 3.12 를 따로 설치할 수 있습니다.
choice /c YN /m " 지금 Python 3.12 를 자동으로 설치할까요"
if errorlevel 2 goto manual
where pymanager >nul 2>nul
if errorlevel 1 goto dlpython
set "MGRTRIED=1"
echo.
echo  Python install manager 로 3.12 를 설치합니다...
pymanager install 3.12
echo.
goto findpy

:dlpython
set "DLTRIED=1"
set "PYURL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
set "PYEXE=%TEMP%\python-3.12.10-amd64.exe"
echo.
echo  python.org 에서 Python 3.12 설치 파일을 받는 중입니다. (약 26MB)
curl.exe -L --fail -o "%PYEXE%" "%PYURL%"
if not errorlevel 1 goto checkdl
powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing -Uri $env:PYURL -OutFile $env:PYEXE"
if errorlevel 1 goto manual
:checkdl
set "PYSIZE="
for %%A in ("%PYEXE%") do set "PYSIZE=%%~zA"
if not defined PYSIZE goto manual
if %PYSIZE% LSS 20000000 goto manual
echo  Python 3.12 설치 중입니다. 1~3분 걸립니다. 다른 Python 과 PATH 는 건드리지 않습니다.
start "" /wait "%PYEXE%" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 Shortcuts=0 AssociateFiles=0
del "%PYEXE%" >nul 2>nul
echo.
goto findpy

:manual
echo  아래 주소에서 Python 3.12 를 설치한 뒤 이 파일을 다시 실행하세요.
echo  https://www.python.org/downloads/release/python-31210/
echo  "Windows installer (64-bit)" 를 받아 설치하고, 설치할 때 "Add python.exe to PATH" 를 체크하세요.
pause
exit /b 1

:fail
echo.
echo  [오류] 설치 중 문제가 생겼습니다. 인터넷 연결을 확인하고 다시 실행해 주세요.
echo  계속 안 되면 이 창의 내용을 캡처해서 보내주세요.
pause
exit /b 1

rem ---- 맞는 Python 찾기 (찾으면 PY 에 실행 명령을 넣는다) ----
:trypy
if defined PY exit /b 0
py -%1 -c "import sys" <nul >nul 2>nul
if errorlevel 1 exit /b 0
set "PY=py -%1"
exit /b 0

:trypath
if defined PY exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys" <nul >nul 2>nul
if errorlevel 1 exit /b 0
set PY="%~1"
exit /b 0

:trycmd
if defined PY exit /b 0
python -c "import sys; sys.exit(0 if (3,9)<=sys.version_info[:2]<=(3,13) else 1)" <nul >nul 2>nul
if errorlevel 1 exit /b 0
set "PY=python"
exit /b 0

:trymgr
if defined PY exit /b 0
if not defined MGRTRIED exit /b 0
pymanager exec -V:3.12 -c "import sys" <nul >nul 2>nul
if errorlevel 1 exit /b 0
set "PY=pymanager exec -V:3.12"
exit /b 0
