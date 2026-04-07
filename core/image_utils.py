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

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "ratio": ratio,
        "quality": quality,
        "prompt": prompt,
        "filename": filename,
        "size": list(image.size),
    }
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return str(img_path), str(meta_path)


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


def save_video(video_bytes: bytes, model_name: str, prompt: str) -> str:
    """
    영상 바이트 데이터를 outputs/YYYY-MM-DD/ 경로에 저장하고 파일 경로를 반환합니다.
    """
    out_dir = get_output_dir()
    timestamp = datetime.now().strftime("%H%M%S_%f")[:13]
    safe_model = model_name.replace(" ", "_").replace("/", "_")
    filename = f"{timestamp}_{safe_model}_video.mp4"
    video_path = out_dir / filename

    with open(str(video_path), "wb") as f:
        f.write(video_bytes)

    return str(video_path)


def load_metadata(meta_path: str) -> dict:
    """메타데이터 JSON 파일을 읽어 반환합니다."""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
