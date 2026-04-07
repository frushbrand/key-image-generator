#!/bin/bash
# ============================================================
#  AI 키 이미지 생성 툴 — macOS / Linux 원클릭 설치 & 실행 스크립트
# ============================================================

set -e

echo ""
echo "========================================"
echo " AI 키 이미지 생성 툴 설치 시작"
echo "========================================"
echo ""

# Python 버전 확인
if ! command -v python3 &>/dev/null; then
    echo "❌ Python3 가 설치되어 있지 않습니다."
    echo "   https://www.python.org/downloads/ 에서 Python 3.10 이상을 설치해 주세요."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED="3.10"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "✅ Python $PYTHON_VERSION 확인 완료"
else
    echo "❌ Python $PYTHON_VERSION 은 지원하지 않습니다. Python 3.10 이상이 필요합니다."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 생성 (없을 경우에만)
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 가상환경 생성 중..."
    python3 -m venv venv
    echo "✅ 가상환경 생성 완료"
fi

# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
echo ""
echo "📥 필요한 패키지 설치 중... (최초 실행 시 시간이 걸릴 수 있습니다)"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✅ 패키지 설치 완료"

# outputs 폴더 생성
mkdir -p outputs

echo ""
echo "========================================"
echo " 🚀 앱을 실행합니다!"
echo " 브라우저에서 http://localhost:7860 으로 접속하세요."
echo " 종료하려면 이 창에서 Ctrl+C 를 누르세요."
echo "========================================"
echo ""

python app.py
