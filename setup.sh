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

# Linux에서 Pillow 컴파일에 필요한 시스템 라이브러리 확인 및 설치
if [[ "$(uname -s)" == "Linux" ]]; then
    echo ""
    echo "🔍 Pillow 컴파일에 필요한 시스템 라이브러리 확인 중..."
    if command -v apt-get &>/dev/null; then
        MISSING_PKGS=()
        for pkg in zlib1g-dev libjpeg-dev libpng-dev libfreetype6-dev; do
            if ! dpkg -s "$pkg" &>/dev/null; then
                MISSING_PKGS+=("$pkg")
            fi
        done
        if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
            echo "📦 누락된 시스템 라이브러리 설치 중: ${MISSING_PKGS[*]}"
            sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends "${MISSING_PKGS[@]}"
            echo "✅ 시스템 라이브러리 설치 완료"
        else
            echo "✅ 필요한 시스템 라이브러리가 모두 설치되어 있습니다."
        fi
    else
        echo "⚠️ apt-get 을 찾을 수 없습니다. Pillow 빌드에 필요한 다음 패키지를 수동으로 설치해 주세요:"
        echo "   zlib, libjpeg, libpng, freetype"
        echo "   (예: sudo yum install zlib-devel libjpeg-devel libpng-devel freetype-devel)"
    fi
fi

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
