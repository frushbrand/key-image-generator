# 화질 설정
QUALITY_OPTIONS = {
    "Standard": {"width_multiplier": 1.0, "description": "기본 화질"},
    "HD": {"width_multiplier": 1.5, "description": "고화질"},
    "Ultra HD": {"width_multiplier": 2.0, "description": "초고화질"},
}

# 화면 비율 설정 (width, height 기준 픽셀, 나노 바나나 2 기준)
ASPECT_RATIOS = {
    "1:1":  {"width": 1024, "height": 1024, "imagen_ratio": "1:1"},
    "16:9": {"width": 1408, "height": 792,  "imagen_ratio": "16:9"},
    "9:16": {"width": 792,  "height": 1408, "imagen_ratio": "9:16"},
    "4:3":  {"width": 1152, "height": 864,  "imagen_ratio": "4:3"},
    "3:4":  {"width": 864,  "height": 1152, "imagen_ratio": "3:4"},
    "2:3":  {"width": 832,  "height": 1248, "imagen_ratio": "2:3"},
    "3:2":  {"width": 1248, "height": 832,  "imagen_ratio": "3:2"},
}

# 모델 정의
MODELS = {
    "나노 바나나 2": {
        "api_name": "gemini-flash-3.1-preview",
        "description": "Gemini 3.1 Flash Image Preview (빠름)",
        "supports_reference_image": True,
    },
    "나노 바나나 프로": {
        "api_name": "gemini-3.0-pro",
        "description": "Gemini 3 Pro 이미지 모델 (고품질)",
        "supports_reference_image": True,
    },
}

# ── Kling 비디오 모델 ──────────────────────────────────────────────────────────

KLING_MODELS = {
    "Kling 3 Standard": {
        "api_name": "kling-v3",
        "description": "Kling v3 Standard (빠름, 범용)",
        "mode": "standard",
    },
    "Kling 3 Pro": {
        "api_name": "kling-v3-pro",
        "description": "Kling v3 Pro (고품질, 고해상도)",
        "mode": "professional",
    },
    "Kling 3 Omni": {
        "api_name": "kling-v3-omni",
        "description": "Kling v3 Omni (네이티브 오디오, 최고 품질)",
        "mode": "professional",
    },
}

KLING_DURATIONS = list(range(3, 16))  # 3 ~ 15초 (1초 단위)
KLING_DEFAULT_DURATION = 5
KLING_DEFAULT_MODEL = "Kling 3 Standard"

KLING_VIDEO_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
KLING_DEFAULT_RATIO = "16:9"

# ── 기본값 ─────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "나노 바나나 2"
DEFAULT_RATIO = "16:9"
DEFAULT_QUALITY = "HD"
DEFAULT_COUNT = 1
MAX_COUNT = 20
MAX_REFERENCE_IMAGES = 10
MAX_RETRY = 3

# 출력 경로
OUTPUT_BASE_DIR = "outputs"

# 설정 파일 경로
SETTINGS_FILE = ".app_settings.json"
