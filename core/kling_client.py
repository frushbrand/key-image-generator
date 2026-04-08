"""
Kling AI 공식 API 클라이언트

Kling v3 / v3 Pro / v3 Omni 모델의 이미지→영상(image-to-video) 생성을 지원합니다.

인증 방식: Access Key + Secret Key → JWT (HS256)
공식 문서: https://kling.ai/document-api
"""

import base64
import io
import time
from pathlib import Path
from typing import Optional

import jwt
import requests
from PIL import Image

KLING_API_BASE = "https://api-singapore.klingai.com"


def _make_jwt(access_key: str, secret_key: str) -> str:
    """Kling API 인증용 JWT 토큰을 생성합니다."""
    now = int(time.time())
    payload = {
        "iss": access_key,
        "exp": now + 1800,  # 30분 유효
        "nbf": now - 5,
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")


def _auth_headers(access_key: str, secret_key: str) -> dict:
    token = _make_jwt(access_key, secret_key)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _image_to_base64(image: Image.Image) -> str:
    """PIL Image를 JPEG base64 문자열로 변환합니다."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _video_to_base64(video_path: str) -> str:
    """비디오 파일을 base64 문자열로 변환합니다."""
    resolved = Path(video_path).resolve()
    if not resolved.is_file():
        raise ValueError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    max_size = 100 * 1024 * 1024  # 100 MB
    if resolved.stat().st_size > max_size:
        raise ValueError("영상 파일 크기가 너무 큽니다. 100 MB 이하의 파일만 지원합니다.")
    with open(resolved, "rb") as f:
        return base64.b64encode(f.read()).decode()


def validate_kling_keys(access_key: str, secret_key: str) -> tuple[bool, str]:
    """
    Kling API Access Key / Secret Key 유효성을 검증합니다.
    반환: (is_valid, message)
    """
    if not access_key or not access_key.strip():
        return False, "Access Key를 입력해주세요."
    if not secret_key or not secret_key.strip():
        return False, "Secret Key를 입력해주세요."
    try:
        _make_jwt(access_key.strip(), secret_key.strip())
    except Exception as e:
        return False, f"❌ JWT 생성 오류: {e}"
    try:
        resp = requests.get(
            f"{KLING_API_BASE}/v1/videos/image2video",
            headers=_auth_headers(access_key.strip(), secret_key.strip()),
            timeout=10,
        )
        if resp.status_code == 401:
            return False, "❌ 유효하지 않은 API 키입니다. Access Key / Secret Key를 확인해주세요."
        # 200 또는 400(파라미터 없음) → 인증은 성공
        return True, "✅ Kling API 키가 유효합니다."
    except requests.exceptions.Timeout:
        return False, "❌ 연결 시간 초과. 네트워크 상태를 확인해주세요."
    except Exception as e:
        return False, f"❌ 연결 오류: {e}"


def create_image_to_video_task(
    access_key: str,
    secret_key: str,
    image: Image.Image,
    prompt: str,
    model: str,
    duration: int,
    aspect_ratio: str,
    mode: str = "standard",
) -> str:
    """
    이미지→영상 생성 작업을 요청하고 task_id를 반환합니다.

    Parameters
    ----------
    access_key : str  Kling Access Key
    secret_key : str  Kling Secret Key
    image      : PIL Image  레퍼런스 이미지
    prompt     : str  영상 방향을 지시하는 프롬프트 (선택)
    model      : str  API 모델명 (예: "kling-v3", "kling-v3-pro")
    duration   : int  영상 길이 (초, 5 또는 10)
    aspect_ratio : str  화면 비율 (예: "16:9")
    mode       : str  "standard" 또는 "professional"
    """
    img_b64 = _image_to_base64(image)

    payload = {
        "model_name": model,
        "image": img_b64,
        "prompt": prompt.strip() if prompt else "",
        "duration": str(duration),
        "aspect_ratio": aspect_ratio,
        "mode": mode,
        "cfg_scale": 0.5,
    }

    resp = requests.post(
        f"{KLING_API_BASE}/v1/videos/image2video",
        headers=_auth_headers(access_key, secret_key),
        json=payload,
        timeout=30,
    )

    if resp.status_code == 401:
        raise PermissionError("Kling API 인증 실패. Access Key / Secret Key를 확인해주세요.")

    resp.raise_for_status()
    data = resp.json()

    code = data.get("code", -1)
    if code != 0:
        raise RuntimeError(f"Kling API 오류 (code={code}): {data.get('message', data)}")

    task_id = data.get("data", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"task_id를 받지 못했습니다. 응답: {data}")

    return task_id


def create_start_end_frame_task(
    access_key: str,
    secret_key: str,
    start_image: Image.Image,
    end_image: Image.Image,
    prompt: str,
    model: str,
    duration: int,
    aspect_ratio: str,
    mode: str = "standard",
) -> str:
    """
    시작-끝 프레임 방식으로 영상 생성 작업을 요청하고 task_id를 반환합니다.

    Parameters
    ----------
    start_image : PIL Image  시작 프레임
    end_image   : PIL Image  끝 프레임
    """
    start_b64 = _image_to_base64(start_image)
    end_b64 = _image_to_base64(end_image)

    payload = {
        "model_name": model,
        "image": start_b64,
        "image_tail": end_b64,
        "prompt": prompt.strip() if prompt else "",
        "duration": str(duration),
        "aspect_ratio": aspect_ratio,
        "mode": mode,
        "cfg_scale": 0.5,
    }

    resp = requests.post(
        f"{KLING_API_BASE}/v1/videos/image2video",
        headers=_auth_headers(access_key, secret_key),
        json=payload,
        timeout=30,
    )

    if resp.status_code == 401:
        raise PermissionError("Kling API 인증 실패. Access Key / Secret Key를 확인해주세요.")

    resp.raise_for_status()
    data = resp.json()

    code = data.get("code", -1)
    if code != 0:
        raise RuntimeError(f"Kling API 오류 (code={code}): {data.get('message', data)}")

    task_id = data.get("data", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"task_id를 받지 못했습니다. 응답: {data}")

    return task_id


def create_video_reference_task(
    access_key: str,
    secret_key: str,
    video_path: str,
    prompt: str,
    model: str,
    duration: int,
    aspect_ratio: str,
    mode: str = "standard",
) -> str:
    """
    레퍼런스 영상을 사용한 영상 생성 작업을 요청하고 task_id를 반환합니다.

    Parameters
    ----------
    video_path : str  로컬 영상 파일 경로 (3~10초)
    """
    video_b64 = _video_to_base64(video_path)

    payload = {
        "model_name": model,
        "video": video_b64,
        "prompt": prompt.strip() if prompt else "",
        "duration": str(duration),
        "aspect_ratio": aspect_ratio,
        "mode": mode,
        "cfg_scale": 0.5,
    }

    resp = requests.post(
        f"{KLING_API_BASE}/v1/videos/video2video",
        headers=_auth_headers(access_key, secret_key),
        json=payload,
        timeout=60,
    )

    if resp.status_code == 401:
        raise PermissionError("Kling API 인증 실패. Access Key / Secret Key를 확인해주세요.")

    resp.raise_for_status()
    data = resp.json()

    code = data.get("code", -1)
    if code != 0:
        raise RuntimeError(f"Kling API 오류 (code={code}): {data.get('message', data)}")

    task_id = data.get("data", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"task_id를 받지 못했습니다. 응답: {data}")

    return task_id


def poll_task_result(
    access_key: str,
    secret_key: str,
    task_id: str,
    endpoint: str = "image2video",
    timeout: int = 300,
    poll_interval: int = 5,
    progress_callback=None,
) -> str:
    """
    영상 생성 작업이 완료될 때까지 폴링하고 영상 URL을 반환합니다.

    Parameters
    ----------
    endpoint         : str  API 엔드포인트 종류 ("image2video" 또는 "video2video")
    progress_callback : callable(elapsed_sec, status_msg)  진행 상황 콜백 (선택)
    """
    deadline = time.time() + timeout
    start = time.time()

    while time.time() < deadline:
        elapsed = int(time.time() - start)
        resp = requests.get(
            f"{KLING_API_BASE}/v1/videos/{endpoint}/{task_id}",
            headers=_auth_headers(access_key, secret_key),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        code = data.get("code", -1)
        if code != 0:
            raise RuntimeError(f"Kling API 폴링 오류 (code={code}): {data.get('message', data)}")

        task = data.get("data", {})
        status = task.get("task_status", "")

        if progress_callback:
            progress_callback(elapsed, status)

        if status == "succeed":
            videos = task.get("task_result", {}).get("videos", [])
            if videos:
                return videos[0]["url"]
            raise RuntimeError("영상 URL을 찾을 수 없습니다.")

        if status == "failed":
            msg = task.get("task_status_msg") or task.get("task_result", {}).get("message", "알 수 없는 오류")
            raise RuntimeError(f"영상 생성 실패: {msg}")

        time.sleep(poll_interval)

    raise TimeoutError(f"영상 생성 타임아웃 ({timeout}초 초과). 나중에 다시 시도해주세요.")


def download_video(video_url: str) -> bytes:
    """영상 URL에서 바이트 데이터를 다운로드합니다."""
    resp = requests.get(video_url, timeout=120)
    resp.raise_for_status()
    return resp.content
