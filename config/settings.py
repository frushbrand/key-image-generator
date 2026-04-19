# 화질 설정 (Gemini API image_size 값 기준)
QUALITY_OPTIONS = {
    "512": {"api_image_size": "512"},
    "1K":  {"api_image_size": "1K"},
    "2K":  {"api_image_size": "2K"},
    "4K":  {"api_image_size": "4K"},
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
        "api_name": "gemini-3.1-flash-image-preview",
        "description": "Gemini 3.1 Flash Image 프리뷰 (속도·대량 작업에 최적화)",
        "supports_reference_image": True,
    },
    "나노 바나나 프로": {
        "api_name": "gemini-3-pro-image-preview",
        "description": "Gemini 3 Pro Image 프리뷰 (전문 애셋 제작, 고급 추론)",
        "supports_reference_image": True,
    },
    "나노 바나나": {
        "api_name": "gemini-2.5-flash-image",
        "description": "Gemini 2.5 Flash Image (빠른 속도·낮은 지연 시간)",
        "supports_reference_image": True,
    },
}

# ── Kling 비디오 모델 ──────────────────────────────────────────────────────────

KLING_MODELS = {
    "Kling v3": {
        "api_name": "kling-v3",
        "description": "Kling v3 (범용, 빠름)",
        "mode": "std",
        "supports_video_reference": False,
    },
    "Kling 3 Omni": {
        "api_name": "kling-v3-omni",
        "description": "Kling v3 Omni (최고 품질, 네이티브 오디오)",
        "mode": "pro",
        "supports_video_reference": True,
    },
}

KLING_DURATIONS = list(range(3, 16))  # 3 ~ 15초 (1초 단위)
KLING_DEFAULT_DURATION = 5
KLING_DEFAULT_MODEL = "Kling v3"

# 클링 화질 옵션 (API mode 값으로 매핑)
KLING_QUALITY_OPTIONS = {
    "720p (Standard)": "std",
    "1080p (Professional)": "pro",
}
KLING_DEFAULT_QUALITY = "720p (Standard)"

KLING_VIDEO_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
KLING_DEFAULT_RATIO = "16:9"

# ── Seedance 비디오 모델 (fal.ai REST API 사용) ────────────────────────────────

SEEDANCE_MODELS = {
    "Seedance 2.0": {
        "api_name": "bytedance/seedance-2.0",
        "description": "Seedance 2.0 (고품질)",
    },
    "Seedance 2.0 Fast": {
        "api_name": "bytedance/seedance-2.0/fast",
        "description": "Seedance 2.0 Fast (빠른 생성, 약간 낮은 품질)",
    },
}

SEEDANCE_DURATIONS = list(range(4, 16))  # 4 ~ 15초 (1초 단위)
SEEDANCE_DEFAULT_DURATION = 5
SEEDANCE_DEFAULT_MODEL = "Seedance 2.0"

SEEDANCE_QUALITY_OPTIONS = {
    "720p": "720p",
    "1080p": "1080p",
    "480p": "480p",
}
SEEDANCE_DEFAULT_QUALITY = "720p"

SEEDANCE_VIDEO_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
SEEDANCE_DEFAULT_RATIO = "16:9"

# ── 기본값 ─────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "나노 바나나 2"
DEFAULT_RATIO = "16:9"
DEFAULT_QUALITY = "2K"
DEFAULT_COUNT = 1
MAX_COUNT = 20
MAX_REFERENCE_IMAGES = 10
MAX_RETRY = 3

# 출력 경로
OUTPUT_BASE_DIR = "outputs"

# 설정 파일 경로
SETTINGS_FILE = ".app_settings.json"
