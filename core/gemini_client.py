"""
Gemini API 클라이언트 모듈
나노 바나나 2 (Gemini 3.1 Flash Image Preview) 및 나노 바나나 프로 (Gemini 3 Pro 이미지 모델)를 지원합니다.
"""

import base64
import io
import time
import concurrent.futures
from typing import Optional

import google.generativeai as genai
from google.generativeai import types
from PIL import Image

from config.settings import (
    MODELS,
    ASPECT_RATIOS,
    QUALITY_OPTIONS,
    MAX_RETRY,
)


def configure_api(api_key: str) -> None:
    """Gemini API 키를 설정합니다."""
    genai.configure(api_key=api_key)


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """API 키 유효성을 검증합니다."""
    if not api_key or not api_key.strip():
        return False, "API 키를 입력해주세요."
    try:
        configure_api(api_key.strip())
        # 간단한 호출로 키 유효성 확인
        client = genai.GenerativeModel("gemini-2.0-flash")
        client.generate_content("test")
        return True, "✅ API 키가 유효합니다."
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "invalid" in err.lower():
            return False, "❌ 유효하지 않은 API 키입니다."
        if "quota" in err.lower():
            return True, "✅ API 키가 유효합니다. (할당량 초과 상태일 수 있습니다)"
        return False, f"❌ 오류 발생: {err}"


def _pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def generate_with_nano_banana_2(
    api_key: str,
    prompt: str,
    ratio: str,
    quality: str,
    reference_images: Optional[list[Image.Image]] = None,
) -> Image.Image:
    """
    나노 바나나 2 (Gemini 3.1 Flash Image Preview) 모델로 이미지를 생성합니다.
    레퍼런스 이미지를 지원합니다.
    """
    configure_api(api_key)

    ratio_cfg = ASPECT_RATIOS[ratio]
    quality_cfg = QUALITY_OPTIONS[quality]
    width = int(ratio_cfg["width"] * quality_cfg["width_multiplier"])
    height = int(ratio_cfg["height"] * quality_cfg["width_multiplier"])

    # 프롬프트에 크기 힌트 추가
    full_prompt = (
        f"{prompt}\n\n"
        f"Image should be {width}x{height} pixels, aspect ratio {ratio}."
    )

    parts = []

    # 레퍼런스 이미지 추가
    if reference_images:
        parts.append("Reference images are provided below. Use them as style/composition reference:\n")
        for img in reference_images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            parts.append(
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
            )
        parts.append("\nNow generate the requested image:\n")

    parts.append(full_prompt)

    model = genai.GenerativeModel(
        model_name=MODELS["나노 바나나 2"]["api_name"],
        generation_config=types.GenerationConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    response = model.generate_content(parts)

    # 응답에서 이미지 추출
    for part in response.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            img_bytes = part.inline_data.data
            if isinstance(img_bytes, str):
                img_bytes = base64.b64decode(img_bytes)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    raise ValueError("모델이 이미지를 반환하지 않았습니다. 프롬프트를 수정하거나 다시 시도해주세요.")


def generate_with_nano_banana_pro(
    api_key: str,
    prompt: str,
    ratio: str,
    quality: str,
    reference_images: Optional[list[Image.Image]] = None,
) -> Image.Image:
    """
    나노 바나나 프로 (Gemini 3 Pro 이미지 모델)로 이미지를 생성합니다.
    레퍼런스 이미지를 지원합니다.
    """
    configure_api(api_key)

    ratio_cfg = ASPECT_RATIOS[ratio]
    quality_cfg = QUALITY_OPTIONS[quality]
    width = int(ratio_cfg["width"] * quality_cfg["width_multiplier"])
    height = int(ratio_cfg["height"] * quality_cfg["width_multiplier"])

    full_prompt = (
        f"{prompt}\n\n"
        f"Image should be {width}x{height} pixels, aspect ratio {ratio}."
    )

    parts = []

    if reference_images:
        parts.append("Reference images are provided below. Use them as style/composition reference:\n")
        for img in reference_images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            parts.append(
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
            )
        parts.append("\nNow generate the requested image:\n")

    parts.append(full_prompt)

    model = genai.GenerativeModel(
        model_name=MODELS["나노 바나나 프로"]["api_name"],
        generation_config=types.GenerationConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    response = model.generate_content(parts)

    for part in response.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            img_bytes = part.inline_data.data
            if isinstance(img_bytes, str):
                img_bytes = base64.b64decode(img_bytes)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    raise ValueError("모델이 이미지를 반환하지 않았습니다. 프롬프트를 수정하거나 다시 시도해주세요.")


def generate_single_image(
    api_key: str,
    model_name: str,
    prompt: str,
    ratio: str,
    quality: str,
    reference_images: Optional[list[Image.Image]] = None,
) -> Image.Image:
    """단일 이미지를 생성합니다. 실패 시 최대 MAX_RETRY회 재시도합니다."""
    last_error = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            if model_name == "나노 바나나 2":
                return generate_with_nano_banana_2(
                    api_key, prompt, ratio, quality, reference_images
                )
            elif model_name == "나노 바나나 프로":
                return generate_with_nano_banana_pro(
                    api_key, prompt, ratio, quality, reference_images
                )
            else:
                raise ValueError(f"알 수 없는 모델: {model_name}")
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRY:
                time.sleep(2 ** attempt)  # 지수 백오프
    raise RuntimeError(f"이미지 생성 실패 ({MAX_RETRY}회 시도): {last_error}")


def generate_batch_images(
    api_key: str,
    model_name: str,
    prompt: str,
    ratio: str,
    quality: str,
    count: int,
    reference_images: Optional[list[Image.Image]] = None,
    progress_callback=None,
) -> list[tuple[Image.Image | None, str]]:
    """
    여러 장의 이미지를 병렬로 생성합니다.
    반환값: [(image_or_None, status_message), ...]
    """
    results: list[tuple[Image.Image | None, str]] = [None] * count

    def _task(idx: int):
        try:
            img = generate_single_image(
                api_key, model_name, prompt, ratio, quality, reference_images
            )
            results[idx] = (img, "success")
            if progress_callback:
                progress_callback(idx, img, None)
        except Exception as e:
            results[idx] = (None, str(e))
            if progress_callback:
                progress_callback(idx, None, str(e))

    max_workers = min(count, 5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_task, i) for i in range(count)]
        concurrent.futures.wait(futures)

    return results
