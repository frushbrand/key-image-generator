"""
AI 영상 키 이미지 생성 툴 - 메인 진입점
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    # .env 파일 자동 로드
    load_dotenv()
except ImportError:
    pass  # python-dotenv 없이도 동작 가능 (환경변수를 직접 설정한 경우)

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ui.components import build_ui
except (ImportError, ModuleNotFoundError) as e:
    print(
        "\n❌ 필수 패키지가 설치되어 있지 않습니다.\n"
        f"   누락된 패키지: {e}\n\n"
        "   아래 방법 중 하나로 설치해 주세요:\n"
        "   • Windows  : setup.bat 파일을 더블클릭\n"
        "   • 직접 설치 : pip install -r requirements.txt\n"
    )
    raise SystemExit(1)


def main():
    # GitHub Codespaces에서는 브라우저 자동 열기 불필요 (포트 포워딩으로 자동 접속)
    in_codespaces = os.environ.get("CODESPACES") == "true"

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=not in_codespaces,
    )


if __name__ == "__main__":
    main()
