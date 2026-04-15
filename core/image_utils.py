"""
이미지 전처리/후처리 유틸리티 모듈
"""

import io
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

from config.settings import OUTPUT_BASE_DIR, ASPECT_RATIOS, QUALITY_OPTIONS

# 갤러리 표시용 썸네일 최대 크기 (픽셀)
THUMBNAIL_SIZE = 512

# 플레이스홀더 이미지 크기 및 색상 (생성 대기 중 슬롯 표시용)
_PLACEHOLDER_SIZE = 512
_PLACEHOLDER_BG_COLOR = (22, 22, 38)
_PLACEHOLDER_TEXT_COLOR = (120, 120, 200)

# 생성 대기 중 슬롯에 표시할 플레이스홀더 이미지 경로
PLACEHOLDER_IMAGE_PATH = str(Path(OUTPUT_BASE_DIR) / "_placeholder.png")


def ensure_placeholder_image() -> str:
    """로딩 플레이스홀더 이미지를 생성합니다 (없는 경우).
    outputs/ 디렉토리에 저장되며 갤러리에서 Gradio가 직접 서빙합니다."""
    path = PLACEHOLDER_IMAGE_PATH
    if not os.path.exists(path):
        Path(OUTPUT_BASE_DIR).mkdir(parents=True, exist_ok=True)
        from PIL import ImageDraw
        img = Image.new("RGB", (_PLACEHOLDER_SIZE, _PLACEHOLDER_SIZE), color=_PLACEHOLDER_BG_COLOR)
        draw = ImageDraw.Draw(img)
        # 가운데에 연한 텍스트 (기본 폰트 사용)
        draw.text((180, 230), "Generating...", fill=_PLACEHOLDER_TEXT_COLOR)
        img.save(path, format="PNG")
    return path


def get_thumbnail_path(image_path: str) -> str:
    """원본 이미지 경로로부터 썸네일 파일 경로를 계산합니다.
    썸네일은 원본과 같은 날짜 디렉토리 내 thumbs/ 서브폴더에 저장됩니다."""
    p = Path(image_path)
    thumb_dir = p.parent / "thumbs"
    return str(thumb_dir / p.name)


def get_video_thumbnail_path(video_path: str) -> str:
    """영상 경로로부터 썸네일 파일 경로를 계산합니다.
    썸네일은 영상과 같은 날짜 디렉토리 내 thumbs/ 서브폴더에 PNG로 저장됩니다."""
    p = Path(video_path)
    thumb_dir = p.parent / "thumbs"
    return str(thumb_dir / (p.stem + ".png"))


def create_video_thumbnail(video_path: str) -> str:
    """영상의 첫 프레임을 추출하여 thumbs/ 서브폴더에 PNG 썸네일로 저장합니다.
    cv2(OpenCV) 또는 ffmpeg가 설치된 경우 실제 프레임을 추출하고,
    그렇지 않으면 플레이스홀더 이미지를 썸네일로 사용합니다."""
    import shutil as _shutil
    import subprocess

    output_base = Path(OUTPUT_BASE_DIR).resolve()
    resolved = Path(video_path).resolve()
    if not str(resolved).startswith(str(output_base) + os.sep):
        raise ValueError(f"영상 경로가 출력 디렉토리 밖에 있습니다: {video_path}")

    # validated resolved 경로에서 thumb_path 계산 (경로 주입 방지)
    safe_video_path = str(resolved)
    thumb_path = get_video_thumbnail_path(safe_video_path)
    thumb_dir = Path(thumb_path).parent
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # 1) cv2(OpenCV) 로 첫 프레임 추출
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(safe_video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS)
            img.save(thumb_path, format="PNG")
            return thumb_path
    except Exception:
        pass

    # 2) ffmpeg CLI 로 첫 프레임 추출
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", safe_video_path,
                "-vframes", "1", "-q:v", "2",
                "-vf", f"scale='min({THUMBNAIL_SIZE},iw)':-1",
                "-y", thumb_path,
            ],
            capture_output=True,
            timeout=15,
        )
        if result.returncode == 0 and Path(thumb_path).exists():
            return thumb_path
    except Exception:
        pass

    # 3) 폴백: 플레이스홀더 이미지를 썸네일로 복사
    if os.path.exists(PLACEHOLDER_IMAGE_PATH):
        _shutil.copy2(PLACEHOLDER_IMAGE_PATH, thumb_path)
    return thumb_path


def create_thumbnail(image_path: str) -> str:
    """원본 이미지를 512px 썸네일로 변환하여 thumbs/ 폴더에 저장하고 경로를 반환합니다."""
    # 경로가 출력 디렉토리 내에 있는지 확인 (경로 주입 방지)
    output_base = Path(OUTPUT_BASE_DIR).resolve()
    resolved = Path(image_path).resolve()
    if not str(resolved).startswith(str(output_base) + os.sep):
        raise ValueError(f"이미지 경로가 출력 디렉토리 밖에 있습니다: {image_path}")
    thumb_path = get_thumbnail_path(image_path)
    Path(thumb_path).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(resolved) as img:
        img = img.convert("RGB")
        img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS)
        img.save(thumb_path, format="PNG")
    return thumb_path


def resize_reference_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """레퍼런스 이미지를 최대 크기 이내로 리사이즈합니다."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    if w >= h:
        new_w, new_h = max_size, int(h * max_size / w)
    else:
        new_w, new_h = int(w * max_size / h), max_size
    return image.resize((new_w, new_h), Image.LANCZOS)


def load_reference_images(file_paths: list[str]) -> list[Image.Image]:
    """
    업로드된 레퍼런스 이미지 파일 경로 목록을 PIL Image 목록으로 변환합니다.
    """
    images = []
    for path in file_paths:
        if path is None:
            continue
        img = Image.open(path).convert("RGB")
        img = resize_reference_image(img)
        images.append(img)
    return images


def get_output_dir() -> Path:
    """오늘 날짜 기반 출력 디렉토리를 반환합니다. 없으면 생성합니다."""
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(OUTPUT_BASE_DIR) / today
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_image(
    image: Image.Image,
    model_name: str,
    ratio: str,
    prompt: str,
    quality: str,
    reference_image_paths: Optional[list] = None,
) -> tuple[str, str]:
    """
    이미지를 outputs/YYYY-MM-DD/ 경로에 저장하고 메타데이터 JSON을 함께 저장합니다.
    반환값: (image_path, metadata_path)
    """
    out_dir = get_output_dir()
    timestamp = datetime.now().strftime("%H%M%S_%f")[:13]

    safe_model = model_name.replace(" ", "_").replace("/", "_")
    safe_ratio = ratio.replace(":", "x")
    filename = f"{timestamp}_{safe_model}_{safe_ratio}.png"
    meta_filename = f"{timestamp}_{safe_model}_{safe_ratio}.json"

    img_path = out_dir / filename
    meta_path = out_dir / meta_filename

    image.save(str(img_path), format="PNG")

    # 갤러리 표시용 썸네일 생성
    try:
        thumb_path = create_thumbnail(str(img_path))
    except Exception:
        thumb_path = str(img_path)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "ratio": ratio,
        "quality": quality,
        "prompt": prompt,
        "filename": filename,
        "size": list(image.size),
        "reference_image_paths": reference_image_paths or [],
    }
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return str(img_path), str(meta_path), thumb_path


def create_zip_from_paths(image_paths: list[str]) -> Optional[str]:
    """
    이미지 경로 목록을 ZIP으로 묶어 outputs/ 에 저장하고 ZIP 경로를 반환합니다.
    """
    if not image_paths:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = Path(OUTPUT_BASE_DIR) / f"batch_{timestamp}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in image_paths:
            if img_path and os.path.exists(img_path):
                zf.write(img_path, os.path.basename(img_path))
                meta_path = img_path.replace(".png", ".json")
                if os.path.exists(meta_path):
                    zf.write(meta_path, os.path.basename(meta_path))

    return str(zip_path)


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def save_video(video_bytes: bytes, model_name: str, prompt: str) -> tuple[str, str]:
    """
    영상 바이트 데이터를 outputs/YYYY-MM-DD/ 경로에 저장하고
    (video_path, thumbnail_path) 튜플을 반환합니다.
    """
    out_dir = get_output_dir()
    timestamp = datetime.now().strftime("%H%M%S_%f")[:13]
    # 경로 구분자 및 특수 문자 제거하여 안전한 파일명 생성
    safe_model = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
    filename = f"{timestamp}_{safe_model}_video.mp4"
    video_path = out_dir / filename

    with open(video_path, "wb") as f:
        f.write(video_bytes)

    meta_filename = f"{timestamp}_{safe_model}_video.json"
    meta_path = out_dir / meta_filename
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "prompt": prompt,
        "filename": filename,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 썸네일 생성
    try:
        thumb_path = create_video_thumbnail(str(video_path))
    except Exception:
        thumb_path = PLACEHOLDER_IMAGE_PATH

    return str(video_path), thumb_path


def load_metadata(meta_path: str) -> dict:
    """메타데이터 JSON 파일을 읽어 반환합니다."""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_existing_outputs() -> list[dict]:
    """
    outputs/ 디렉토리에 저장된 이미지 파일을 스캔하여 메타데이터와 함께 반환합니다.
    반환값: 각 항목은 {"image": PIL.Image, "image_path": str, "model": str,
                       "ratio": str, "quality": str, "prompt": str} 딕셔너리
    """
    items: list[dict] = []
    out_base = Path(OUTPUT_BASE_DIR)
    if not out_base.exists():
        return items

    # 날짜 폴더를 오래된 순서대로 스캔
    for date_dir in sorted(out_base.iterdir()):
        if not date_dir.is_dir():
            continue
        for img_path in sorted(date_dir.glob("*.png")):
            meta_path = img_path.with_suffix(".json")
            try:
                meta: dict = {}
                if meta_path.exists():
                    meta = load_metadata(str(meta_path))
                # 썸네일 생성 (없으면 새로 만들고, 이미 있으면 그대로 사용)
                thumb_path = get_thumbnail_path(str(img_path))
                if not Path(thumb_path).exists():
                    try:
                        thumb_path = create_thumbnail(str(img_path))
                    except Exception:
                        thumb_path = str(img_path)
                items.append(
                    {
                        # image는 None으로 설정: 시작 시 전체 이미지를 메모리에 로드하지 않아 RAM 절약
                        "image": None,
                        "image_path": str(img_path),
                        "thumbnail_path": thumb_path,
                        "model": meta.get("model", ""),
                        "ratio": meta.get("ratio", ""),
                        "quality": meta.get("quality", ""),
                        "prompt": meta.get("prompt", ""),
                        "reference_image_paths": meta.get("reference_image_paths", []),
                    }
                )
            except Exception:
                continue

    return items


def load_existing_video_outputs() -> list[dict]:
    """
    outputs/ 디렉토리에 저장된 영상 파일을 스캔하여 메타데이터와 함께 반환합니다.
    최신순(날짜 역순, 파일명 역순)으로 정렬하여 반환합니다.
    반환값: 각 항목은 {"path": str, "label": str, "timestamp": str,
                        "model": str, "prompt": str, "date": str} 딕셔너리
    """
    items: list[dict] = []
    out_base = Path(OUTPUT_BASE_DIR)
    if not out_base.exists():
        return items

    for date_dir in sorted(out_base.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for video_path in sorted(date_dir.glob("*.mp4"), reverse=True):
            meta_path = video_path.with_suffix(".json")
            try:
                meta: dict = {}
                if meta_path.exists():
                    meta = load_metadata(str(meta_path))
                model = meta.get("model", "알 수 없음")
                prompt = meta.get("prompt", "")
                short_prompt = prompt[:40] + ("..." if len(prompt) > 40 else "")
                label = f"{date_dir.name} | {model} | {short_prompt}"
                # 썸네일 경로 계산 (없으면 생성 시도)
                thumb_path = get_video_thumbnail_path(str(video_path))
                if not Path(thumb_path).exists():
                    try:
                        thumb_path = create_video_thumbnail(str(video_path))
                    except Exception:
                        thumb_path = PLACEHOLDER_IMAGE_PATH
                items.append(
                    {
                        "path": str(video_path),
                        "thumbnail_path": thumb_path,
                        "label": label,
                        "timestamp": meta.get("timestamp", ""),
                        "model": model,
                        "prompt": prompt,
                        "date": date_dir.name,
                    }
                )
            except Exception:
                continue

    return items


def get_disk_usage_text() -> str:
    """루트 파티션의 전체·사용·여유 디스크 용량 문자열을 반환합니다."""
    import shutil

    try:
        total, used, free = shutil.disk_usage("/")

        def fmt(n: int) -> str:
            if n >= 1024 ** 3:
                return f"{n / 1024 ** 3:.1f} GB"
            return f"{n / 1024 ** 2:.0f} MB"

        return (
            f"💾 저장 공간: 전체 {fmt(total)} | 사용 {fmt(used)} | "
            f"**남은 용량: {fmt(free)}**"
        )
    except Exception:
        return "💾 저장 공간 정보를 가져올 수 없습니다."
