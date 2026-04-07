"""
AI 영상 키 이미지 생성 툴 - 메인 진입점
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# .env 파일 자동 로드
load_dotenv()

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from ui.components import build_ui


def main():
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
