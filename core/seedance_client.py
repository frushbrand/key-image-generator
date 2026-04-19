"""
Seedance 2.0 API 클라이언트 (fal.ai REST API 사용)

Seedance 2.0 / Seedance 2.0 Fast 모델의
- 텍스트→영상 (text-to-video)
- 이미지→영상 (image-to-video, 첫 프레임 모드)
- 시작-끝 프레임 (image-to-video, 두 이미지)
생성을 지원합니다.

인증 방식: API Key → Authorization: Key {api_key}
API 제공사: fal.ai (https://fal.ai)
공식 문서: https://fal.ai/models/bytedance/seedance-2.0
"""

import io
import time
from pathlib import Path

import requests
from PIL import Image

# ── fal.ai 엔드포인트 상수 ────────────────────────────────────────────────────

FAL_QUEUE_BASE = "https://queue.fal.run"
FAL_STORAGE_UPLOAD_URL = "https://api.fal.ai/v1/storage/upload"

# 모델별 fal.ai 엔드포인트 경로 (queue 기반 비동기 생성)
_SEEDANCE_T2V = "bytedance/seedance-2.0/text-to-video"
_SEEDANCE_I2V = "bytedance/seedance-2.0/image-to-video"
_SEEDANCE_FAST_T2V = "bytedance/seedance-2.0/fast/text-to-video"
_SEEDANCE_FAST_I2V = "bytedance/seedance-2.0/fast/image-to-video"

# 모델 이름 → (t2v 경로, i2v 경로) 매핑
_MODEL_ENDPOINTS: dict[str, tuple[str, str]] = {
    "bytedance/seedance-2.0": (_SEEDANCE_T2V, _SEEDANCE_I2V),
    "bytedance/seedance-2.0/fast": (_SEEDANCE_FAST_T2V, _SEEDANCE_FAST_I2V),
}


def _auth_headers(api_key: str) -> dict:
    """fal.ai 인증 헤더를 반환합니다."""
    return {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }


def _upload_image(image: Image.Image, api_key: str) -> str:
    """
    PIL Image를 fal.ai 스토리지에 업로드하고 공개 URL을 반환합니다.

    Parameters
    ----------
    image   : PIL Image  업로드할 이미지
    api_key : str        fal.ai API 키
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    resp = requests.post(
        FAL_STORAGE_UPLOAD_URL,
        headers={"Authorization": f"Key {api_key}"},
        files={"file": ("image.jpg", buf, "image/jpeg")},
        timeout=60,
    )

    if resp.status_code == 401:
        raise PermissionError("Seedance API 인증 실패. API 키를 확인해주세요.")

    if not resp.ok:
        raise RuntimeError(f"이미지 업로드 실패 ({resp.status_code}): {resp.text}")

    data = resp.json()
    url = data.get("url") or data.get("file_url")
    if not url:
        raise RuntimeError(f"업로드 응답에서 URL을 찾을 수 없습니다: {data}")
    return url


def _submit_queue(endpoint: str, api_key: str, payload: dict) -> str:
    """
    fal.ai 큐에 작업을 제출하고 request_id를 반환합니다.

    Parameters
    ----------
    endpoint : str  fal.ai 모델 경로 (예: "bytedance/seedance-2.0/text-to-video")
    api_key  : str  fal.ai API 키
    payload  : dict 생성 파라미터
    """
    url = f"{FAL_QUEUE_BASE}/{endpoint}"
    resp = requests.post(
        url,
        headers=_auth_headers(api_key),
        json=payload,
        timeout=30,
    )

    if resp.status_code == 401:
        raise PermissionError("Seedance API 인증 실패. API 키를 확인해주세요.")

    if not resp.ok:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"Seedance API 오류 ({resp.status_code}): {body}")

    data = resp.json()
    request_id = data.get("request_id")
    if not request_id:
        raise RuntimeError(f"request_id를 받지 못했습니다. 응답: {data}")
    return request_id


def validate_seedance_key(api_key: str) -> tuple[bool, str]:
    """
    fal.ai API 키 유효성을 검증합니다.
    반환: (is_valid, message)
    """
    if not api_key or not api_key.strip():
        return False, "API 키를 입력해주세요."

    try:
        # 존재하지 않는 request_id로 조회 → 401(인증 실패) vs 404(정상 키) 구분
        resp = requests.get(
            f"{FAL_QUEUE_BASE}/{_SEEDANCE_T2V}/requests/key_validation_test",
            headers=_auth_headers(api_key.strip()),
            timeout=10,
        )
        if resp.status_code == 401:
            return False, "❌ 유효하지 않은 API 키입니다. fal.ai API 키를 확인해주세요."
        # 404 = 키는 유효하지만 request_id가 없음 → 정상
        return True, "✅ Seedance (fal.ai) API 키가 유효합니다."
    except requests.exceptions.Timeout:
        return False, "❌ 연결 시간 초과. 네트워크 상태를 확인해주세요."
    except Exception as e:
        return False, f"❌ 연결 오류: {e}"


def create_text_to_video_task(
    api_key: str,
    prompt: str,
    model: str,
    duration: int,
    aspect_ratio: str,
    resolution: str = "720p",
    generate_audio: bool = True,
) -> tuple[str, str]:
    """
    텍스트만으로 영상 생성 작업을 요청하고 (request_id, endpoint) 튜플을 반환합니다.

    Parameters
    ----------
    api_key      : str  fal.ai API 키
    prompt       : str  영상 내용 프롬프트
    model        : str  모델 기본 경로 (예: "bytedance/seedance-2.0")
    duration     : int  영상 길이 (초)
    aspect_ratio : str  화면 비율 (예: "16:9")
    resolution   : str  해상도 ("480p" | "720p" | "1080p")
    generate_audio : bool  네이티브 오디오 생성 여부
    """
    t2v_endpoint, _ = _MODEL_ENDPOINTS.get(model, (_SEEDANCE_T2V, _SEEDANCE_I2V))

    payload = {
        "prompt": prompt.strip() if prompt else "",
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "generate_audio": generate_audio,
    }

    request_id = _submit_queue(t2v_endpoint, api_key, payload)
    return request_id, t2v_endpoint


def create_image_to_video_task(
    api_key: str,
    image: Image.Image,
    prompt: str,
    model: str,
    duration: int,
    aspect_ratio: str,
    resolution: str = "720p",
    generate_audio: bool = True,
) -> tuple[str, str]:
    """
    이미지→영상 생성 작업(첫 프레임 모드)을 요청하고 (request_id, endpoint) 튜플을 반환합니다.

    Parameters
    ----------
    api_key   : str        fal.ai API 키
    image     : PIL Image  시작 프레임으로 사용할 이미지
    """
    _, i2v_endpoint = _MODEL_ENDPOINTS.get(model, (_SEEDANCE_T2V, _SEEDANCE_I2V))

    image_url = _upload_image(image, api_key)

    payload = {
        "image_url": image_url,
        "prompt": prompt.strip() if prompt else "",
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "generate_audio": generate_audio,
    }

    request_id = _submit_queue(i2v_endpoint, api_key, payload)
    return request_id, i2v_endpoint


def create_start_end_frame_task(
    api_key: str,
    start_image: Image.Image,
    end_image: Image.Image,
    prompt: str,
    model: str,
    duration: int,
    aspect_ratio: str,
    resolution: str = "720p",
    generate_audio: bool = True,
) -> tuple[str, str]:
    """
    시작-끝 프레임 방식 영상 생성 작업을 요청하고 (request_id, endpoint) 튜플을 반환합니다.
    두 이미지를 fal.ai 스토리지에 업로드한 뒤 image-to-video 엔드포인트로 전송합니다.

    Parameters
    ----------
    start_image : PIL Image  시작 프레임
    end_image   : PIL Image  끝 프레임
    """
    _, i2v_endpoint = _MODEL_ENDPOINTS.get(model, (_SEEDANCE_T2V, _SEEDANCE_I2V))

    start_url = _upload_image(start_image, api_key)
    end_url = _upload_image(end_image, api_key)

    payload = {
        "image_url": start_url,
        "tail_image_url": end_url,
        "prompt": prompt.strip() if prompt else "",
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "generate_audio": generate_audio,
    }

    request_id = _submit_queue(i2v_endpoint, api_key, payload)
    return request_id, i2v_endpoint


def poll_task_result(
    api_key: str,
    request_id: str,
    endpoint: str,
    timeout: int = 300,
    poll_interval: int = 5,
    progress_callback=None,
) -> str:
    """
    영상 생성 작업이 완료될 때까지 폴링하고 영상 URL을 반환합니다.

    Parameters
    ----------
    api_key           : str  fal.ai API 키
    request_id        : str  작업 요청 ID
    endpoint          : str  모델 엔드포인트 경로
    timeout           : int  최대 대기 시간 (초)
    poll_interval     : int  폴링 간격 (초)
    progress_callback : callable(elapsed_sec, status_msg)  진행 상황 콜백 (선택)
    """
    deadline = time.time() + timeout
    start = time.time()

    status_url = f"{FAL_QUEUE_BASE}/{endpoint}/requests/{request_id}/status"
    result_url = f"{FAL_QUEUE_BASE}/{endpoint}/requests/{request_id}"

    while time.time() < deadline:
        elapsed = int(time.time() - start)

        resp = requests.get(
            status_url,
            headers=_auth_headers(api_key),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "")

        if progress_callback:
            progress_callback(elapsed, status)

        if status == "COMPLETED":
            # 결과 조회
            result_resp = requests.get(
                result_url,
                headers=_auth_headers(api_key),
                timeout=30,
            )
            result_resp.raise_for_status()
            result = result_resp.json()

            # fal.ai 응답: {"video": {"url": "..."}} 또는 {"video_url": "..."}
            video = result.get("video") or {}
            video_url = video.get("url") if isinstance(video, dict) else video
            if not video_url:
                video_url = result.get("video_url") or result.get("url")
            if not video_url:
                raise RuntimeError(f"영상 URL을 찾을 수 없습니다. 응답: {result}")
            return video_url

        if status == "FAILED":
            error = data.get("error") or data.get("detail") or "알 수 없는 오류"
            raise RuntimeError(f"영상 생성 실패: {error}")

        time.sleep(poll_interval)

    raise TimeoutError(f"영상 생성 타임아웃 ({timeout}초 초과). 나중에 다시 시도해주세요.")


def download_video(video_url: str) -> bytes:
    """영상 URL에서 바이트 데이터를 다운로드합니다."""
    resp = requests.get(video_url, timeout=120)
    resp.raise_for_status()
    return resp.content
