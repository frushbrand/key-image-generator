@echo off
chcp 65001 >nul
:: ============================================================
::  AI 키 이미지 생성 툴 — Windows 원클릭 설치 & 실행 스크립트
:: ============================================================

echo.
echo ========================================
echo  AI 키 이미지 생성 툴 설치 시작
echo ========================================
echo.

:: Python 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python 이 설치되어 있지 않습니다.
    echo     https://www.python.org/downloads/ 에서 Python 3.10 이상을 설치해 주세요.
    echo     설치 시 "Add Python to PATH" 옵션을 반드시 체크하세요.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% 확인 완료

cd /d "%~dp0"

:: 가상환경 생성 (없을 경우에만)
if not exist "venv\" (
    echo.
    echo [..] 가상환경 생성 중...
    python -m venv venv
    echo [OK] 가상환경 생성 완료
)

:: 가상환경 활성화
call venv\Scripts\activate.bat

:: 의존성 설치
echo.
echo [..] 필요한 패키지 설치 중... ^(최초 실행 시 시간이 걸릴 수 있습니다^)
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo [OK] 패키지 설치 완료

:: outputs 폴더 생성
if not exist "outputs\" mkdir outputs

echo.
echo ========================================
echo  앱을 실행합니다!
echo  브라우저에서 http://localhost:7860 으로 접속하세요.
echo  종료하려면 이 창을 닫거나 Ctrl+C 를 누르세요.
echo ========================================
echo.

python app.py

pause
