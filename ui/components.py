"""
Gradio UI 컴포넌트 및 이벤트 핸들러 모듈
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from config.settings import (
    MODELS,
    ASPECT_RATIOS,
    QUALITY_OPTIONS,
    DEFAULT_MODEL,
    DEFAULT_RATIO,
    DEFAULT_QUALITY,
    DEFAULT_COUNT,
    MAX_COUNT,
    MAX_REFERENCE_IMAGES,
    SETTINGS_FILE,
    OUTPUT_BASE_DIR,
    KLING_MODELS,
    KLING_DEFAULT_DURATION,
    KLING_DEFAULT_MODEL,
    KLING_VIDEO_RATIOS,
    KLING_DEFAULT_RATIO,
    KLING_QUALITY_OPTIONS,
    KLING_DEFAULT_QUALITY,
)
from core.gemini_client import validate_api_key, generate_batch_images
from core.generation_stats import (
    get_avg_video_time,
    record_video_generation,
)
from core.kling_client import (
    validate_kling_keys,
    create_text_to_video_task,
    create_image_to_video_task,
    create_start_end_frame_task,
    create_video_reference_task,
    poll_task_result,
    download_video,
)
from core.image_utils import (
    load_reference_images,
    save_image,
    save_video,
    create_zip_from_paths,
    get_thumbnail_path,
    get_video_thumbnail_path,
    PLACEHOLDER_IMAGE_PATH,
    ensure_placeholder_image,
    load_existing_outputs,
    load_existing_video_outputs,
    get_disk_usage_text,
)
from ui.gallery import GalleryState, GalleryItem, VideoGalleryState, VideoGalleryItem


# ── 앱 테마 및 CSS (Gradio 6.0+: launch()에 전달) ───────────────────────────

APP_THEME = gr.themes.Soft()

APP_CSS = """
        .title-text { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }
        .subtitle-text { color: #666; margin-bottom: 1rem; }
        .generate-btn { background: #4f46e5 !important; color: white !important; font-size: 1.1rem !important; }

        /* 라이트박스: 닫기(X) 버튼만 표시, 나머지 모두 숨김 */
        /* Gradio 5 */
        .lightbox button,
        [data-testid="lightbox"] button {
            display: none !important;
        }
        .lightbox button:last-child,
        [data-testid="lightbox"] button:last-child {
            display: flex !important;
            width: 48px !important;
            height: 48px !important;
            font-size: 1.4rem !important;
            min-width: 48px !important;
        }
        .lightbox button:last-child svg,
        [data-testid="lightbox"] button:last-child svg {
            width: 28px !important;
            height: 28px !important;
        }
        /* Gradio 6+: 라이트박스는 .preview 클래스 사용
           media-button(이미지 컨테이너)과 thumbnail-item(탐색 썸네일)은 숨기지 않음 */
        .preview button:not([aria-label="Close"]):not([title="Close"]):not(.media-button):not(.thumbnail-item):not(.thumbnail-small) {
            display: none !important;
        }
        .preview button[aria-label="Close"],
        .preview button[title="Close"] {
            display: flex !important;
            width: 48px !important;
            height: 48px !important;
            min-width: 48px !important;
        }
        .preview button[aria-label="Close"] svg,
        .preview button[title="Close"] svg {
            width: 28px !important;
            height: 28px !important;
        }

        /* 호버 플로팅 오버레이 버튼 hover 효과 */
        #gha-ov button:hover {
            background: rgba(40,40,40,0.92) !important;
            transform: scale(1.08);
        }

        /* 공유(Share) 버튼 완전 숨김 */
        button[aria-label="Share"],
        button[title="Share"],
        button[aria-label*="share" i],
        [data-testid="share-button"],
        .share-button,
        .lightbox button[aria-label="Share"],
        [data-testid="lightbox"] button[aria-label="Share"],
        .image-frame button[aria-label="Share"],
        .fixed button[aria-label="Share"],
        [class*="icon-button"][aria-label="Share"],
        [class*="share"] button,
        button[class*="share"] {
            display: none !important;
        }

        /* 레퍼런스 이미지 파일 업로드 헤더 아이콘 버튼 숨김 (업로드 추가·전체 삭제) */
        #ref-upload-main .icon-button-wrapper,
        #ref-upload-add .icon-button-wrapper {
            display: none !important;
        }

        /* 파일 행 내 개별 삭제(×) 버튼은 표시 유지 */
        .label-clear-button {
            opacity: 1 !important;
            visibility: visible !important;
            pointer-events: auto !important;
        }

        /* 갤러리 썸네일 캡션: 2줄로 제한 (프롬프트 미리보기) */
        [data-testid="thumbnail"] [data-testid="caption"],
        [data-testid="thumbnail"] .caption,
        .gallery-item [data-testid="caption"],
        .gallery-item .caption {
            display: -webkit-box !important;
            -webkit-line-clamp: 2 !important;
            -webkit-box-orient: vertical !important;
            overflow: hidden !important;
            white-space: normal !important;
            max-height: 3em !important;
        }

        /* 라이트박스 캡션: 전체 표시 (프롬프트 전체 확인) */
        /* Gradio 5 */
        [data-testid="lightbox"] [data-testid="caption"],
        [data-testid="lightbox"] .caption,
        [data-testid="lightbox"] [class*="caption"],
        .fixed [data-testid="caption"],
        /* Gradio 6+ */
        .preview .caption {
            white-space: pre-wrap !important;
            overflow: visible !important;
            text-overflow: unset !important;
            display: block !important;
            max-height: none !important;
            -webkit-line-clamp: unset !important;
            background: rgba(0,0,0,0.7) !important;
            color: #fff !important;
            padding: 8px 12px !important;
            border-radius: 6px !important;
            margin-top: 6px !important;
            position: relative !important;
            z-index: 5 !important;
        }

        /* 라이트박스 프롬프트 복사 버튼 */
        #lb-copy-prompt-btn {
            position: fixed;
            z-index: 21000;
            background: rgba(30,30,50,0.88);
            color: #fff;
            border: none;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 0.82rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5);
            backdrop-filter: blur(4px);
            transition: background 0.15s, transform 0.1s;
            pointer-events: auto;
            white-space: nowrap;
        }
        #lb-copy-prompt-btn:hover {
            background: rgba(60,60,100,0.95);
            transform: scale(1.05);
        }

        /* 생성 대기 중(pending) 플레이스홀더 이미지: 맥박 애니메이션 */
        [data-testid="thumbnail"] img[src*="_placeholder"],
        .thumbnail-item img[src*="_placeholder"] {
            animation: gallery-pending-pulse 1.4s ease-in-out infinite !important;
            filter: brightness(0.65) !important;
        }
        @keyframes gallery-pending-pulse {
            0%, 100% { opacity: 0.35; }
            50%       { opacity: 0.85; }
        }

        /* 레퍼런스 이미지 상세 패널: 이미지가 잘리지 않도록 */
        #detail-ref-gen img,
        #detail-ref-full img {
            object-fit: contain !important;
            max-height: 400px !important;
        }

        /* 파일 다운로드 출력 위젯: MutationObserver 감지를 위해 DOM에 존재하되 숨김
           Gradio 5는 visible=False 시 DOM에서 완전히 제거(Svelte 조건부 렌더링)하므로
           JS에서 접근해야 하는 컴포넌트는 visible=True + CSS hidden 방식 사용 */
        #single-png-gen, #selected-zip-gen,
        #single-png-gallery, #full-zip-gallery, #selected-zip-gallery,
        #single-video-gen, #selected-videos-zip-gen {
            display: none !important;
        }
        /* 내부 상태/트리거 텍스트박스: display:none 대신 화면 밖 배치 사용
           display:none 시 Gradio 5 Svelte 이벤트 파이프라인이 synthetic event를
           처리하지 않아 Python 핸들러가 트리거되지 않음 */
        #ms-state-gen, #ms-state-gallery, #ms-state-video,
        #overlay-dl-gen, #overlay-vid-gen, #overlay-ref-gen,
        #overlay-dl-gallery, #overlay-vid-gallery, #overlay-ref-gallery,
        #overlay-dl-video,
        #lb-ref-urls-state {
            position: absolute !important;
            left: -9999px !important;
            top: auto !important;
            width: 1px !important;
            height: 1px !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }
        """


# 레퍼런스 이미지 미리보기 썸네일 크기 (픽셀)
_REF_THUMBNAIL_SIZE = 256

# 새로고침 후에도 유지할 레퍼런스 이미지 영구 저장 디렉토리
_PERSISTENT_REFS_DIR = Path(OUTPUT_BASE_DIR) / ".refs"


def _ensure_refs_dir() -> None:
    _PERSISTENT_REFS_DIR.mkdir(parents=True, exist_ok=True)


def _save_ref_images_persistent(file_paths: list) -> list:
    """레퍼런스 이미지를 영구 저장소로 복사하고 영구 경로 목록을 반환합니다."""
    if not file_paths:
        return []
    _ensure_refs_dir()
    persistent_paths = []
    for src in file_paths:
        if not src or not os.path.exists(src):
            continue
        try:
            with open(src, "rb") as fh:
                h = hashlib.sha256(fh.read()).hexdigest()[:16]
            ext = Path(src).suffix or ".png"
            dst = _PERSISTENT_REFS_DIR / f"{h}{ext}"
            if not dst.exists():
                shutil.copy2(src, dst)
            persistent_paths.append(str(dst))
        except Exception:
            continue
    return persistent_paths


def _load_persistent_ref_paths() -> list:
    """설정 파일에서 저장된 레퍼런스 이미지 영구 경로를 로드합니다."""
    cfg = load_settings()
    paths = cfg.get("ref_image_paths", [])
    return [p for p in paths if os.path.exists(p)]


def _save_gen_setting(key: str):
    """단일 생성 설정을 settings 파일에 저장하는 핸들러 팩토리."""
    def handler(value):
        cfg = load_settings()
        cfg[key] = value
        save_settings(cfg)
    return handler


def _save_ref_images_to_settings(files):
    """업로드된 레퍼런스 이미지를 영구 저장하고 경로를 settings에 기록합니다."""
    paths = []
    if files:
        paths = [f if isinstance(f, str) else f.path for f in files if f is not None]
    persistent = _save_ref_images_persistent(paths)
    cfg = load_settings()
    cfg["ref_image_paths"] = persistent
    save_settings(cfg)


# ── 경로 안전 검사 ──────────────────────────────────────────────────────────

def _is_safe_output_path(path: str) -> bool:
    """파일 경로가 반드시 outputs/ 디렉토리 내에 있는지 확인합니다."""
    try:
        output_base = Path(OUTPUT_BASE_DIR).resolve()
        resolved = Path(path).resolve()
        # Python 3.9+ Path.is_relative_to() 대신 호환성 있는 방법 사용
        return str(resolved).startswith(str(output_base) + os.sep) or resolved == output_base
    except Exception:
        return False


# ── 설정 파일 읽기/쓰기 ──────────────────────────────────────────────────────

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # .env 파일 폴백
    env_path = Path(".env")
    settings: dict = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key and key != "your_google_api_key_here":
                    settings["api_key"] = key
            elif line.startswith("KLING_ACCESS_KEY="):
                key = line.split("=", 1)[1].strip()
                if key and key != "your_kling_access_key_here":
                    settings["kling_access_key"] = key
            elif line.startswith("KLING_SECRET_KEY="):
                key = line.split("=", 1)[1].strip()
                if key and key != "your_kling_secret_key_here":
                    settings["kling_secret_key"] = key
    return settings


def save_settings(settings: dict) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


# ── 이벤트 핸들러 ────────────────────────────────────────────────────────────

def on_validate_key(api_key: str):
    ok, msg = validate_api_key(api_key)
    if ok:
        cfg = load_settings()
        cfg["api_key"] = api_key.strip()
        save_settings(cfg)
    return msg


def on_save_key(api_key: str):
    if not api_key or not api_key.strip():
        return "❌ API 키를 입력해주세요."
    cfg = load_settings()
    cfg["api_key"] = api_key.strip()
    save_settings(cfg)
    return "💾 API 키가 저장되었습니다."


def on_validate_kling_keys(access_key: str, secret_key: str):
    ok, msg = validate_kling_keys(access_key, secret_key)
    if ok:
        cfg = load_settings()
        cfg["kling_access_key"] = access_key.strip()
        cfg["kling_secret_key"] = secret_key.strip()
        save_settings(cfg)
    return msg


def on_save_kling_keys(access_key: str, secret_key: str):
    if not access_key or not access_key.strip():
        return "❌ Access Key를 입력해주세요."
    if not secret_key or not secret_key.strip():
        return "❌ Secret Key를 입력해주세요."
    cfg = load_settings()
    cfg["kling_access_key"] = access_key.strip()
    cfg["kling_secret_key"] = secret_key.strip()
    save_settings(cfg)
    return "💾 Kling API 키가 저장되었습니다."


def build_unified_video_fn(gallery_state: GalleryState, video_gallery_state: "VideoGalleryState"):
    """레퍼런스 이미지 / 시작-끝 프레임 / 영상 레퍼런스 / 텍스트 모드를 통합 처리하는 핸들러 팩토리"""

    def generate_video(
        mode: str,           # "image" | "start_end" | "video_ref"
        ref_image,           # 레퍼런스 이미지 모드 (PIL Image or ndarray)
        start_image,         # 시작-끝 프레임 모드: 시작 프레임
        end_image,           # 시작-끝 프레임 모드: 끝 프레임
        ref_video,           # 영상 레퍼런스 모드: 파일 경로 (str)
        prompt: str,
        model_label: str,
        duration: int,
        aspect_ratio: str,
        enable_audio: bool,
        kling_quality: str,  # "720p (Standard)" | "1080p (Professional)"
        progress=gr.Progress(track_tqdm=True),
    ):
        cfg = load_settings()
        ak = cfg.get("kling_access_key", "").strip()
        sk = cfg.get("kling_secret_key", "").strip()
        if not ak or not sk:
            gr.Warning("API 키 설정 탭에서 Kling Access Key와 Secret Key를 먼저 저장해주세요.")
            return None, "❌ Kling API 키 없음"

        model_cfg = KLING_MODELS[model_label]
        api_model = model_cfg["api_name"]
        # 선택한 화질로 모드 결정 (기본값은 모델별 기본 모드)
        mode_api = KLING_QUALITY_OPTIONS.get(kling_quality, model_cfg["mode"])

        # 영상 갤러리에 pending 슬롯 사전 할당
        gallery_index = video_gallery_state.allocate_pending_item(
            model=model_label,
            prompt=prompt or "",
            ratio=aspect_ratio,
            quality=kling_quality,
            duration=int(duration),
        )

        progress(0.05, desc="영상 생성 작업 요청 중...")
        endpoint = "image2video"

        try:
            if mode == "start_end":
                if start_image is None or end_image is None:
                    video_gallery_state.fill_pending_item(gallery_index, "", "", "failed", "시작/끝 프레임 없음")
                    return None, "❌ 시작 프레임과 끝 프레임을 모두 업로드해주세요."
                start_pil = (
                    start_image if isinstance(start_image, Image.Image)
                    else Image.fromarray(start_image).convert("RGB")
                )
                end_pil = (
                    end_image if isinstance(end_image, Image.Image)
                    else Image.fromarray(end_image).convert("RGB")
                )
                task_id = create_start_end_frame_task(
                    access_key=ak, secret_key=sk,
                    start_image=start_pil, end_image=end_pil,
                    prompt=prompt or "", model=api_model,
                    duration=int(duration), aspect_ratio=aspect_ratio,
                    mode=mode_api, enable_audio=enable_audio,
                )
            elif mode == "video_ref":
                if ref_video is None:
                    video_gallery_state.fill_pending_item(gallery_index, "", "", "failed", "레퍼런스 영상 없음")
                    return None, "❌ 레퍼런스 영상을 업로드해주세요."
                video_path = ref_video if isinstance(ref_video, str) else str(ref_video)
                task_id = create_video_reference_task(
                    access_key=ak, secret_key=sk,
                    video_path=video_path,
                    prompt=prompt or "", model=api_model,
                    duration=int(duration), aspect_ratio=aspect_ratio,
                    mode=mode_api, enable_audio=enable_audio,
                )
                endpoint = "video2video"
            else:
                # 레퍼런스 이미지 모드 — 이미지가 없으면 텍스트 투 비디오로 대체
                if ref_image is None:
                    if not (prompt or "").strip():
                        video_gallery_state.fill_pending_item(gallery_index, "", "", "failed", "프롬프트 없음")
                        return None, "❌ 레퍼런스 이미지가 없으면 프롬프트를 입력해주세요."
                    progress(0.05, desc="텍스트→영상 작업 요청 중...")
                    task_id = create_text_to_video_task(
                        access_key=ak, secret_key=sk,
                        prompt=prompt, model=api_model,
                        duration=int(duration), aspect_ratio=aspect_ratio,
                        mode=mode_api, enable_audio=enable_audio,
                    )
                    endpoint = "text2video"
                else:
                    pil_image = (
                        ref_image if isinstance(ref_image, Image.Image)
                        else Image.fromarray(ref_image).convert("RGB")
                    )
                    task_id = create_image_to_video_task(
                        access_key=ak, secret_key=sk,
                        image=pil_image,
                        prompt=prompt or "", model=api_model,
                        duration=int(duration), aspect_ratio=aspect_ratio,
                        mode=mode_api, enable_audio=enable_audio,
                    )
        except Exception as e:
            video_gallery_state.fill_pending_item(gallery_index, "", "", "failed", str(e))
            return None, f"❌ 작업 생성 실패: {e}"

        progress(0.15, desc=f"대기 중... (task_id: {task_id[:8]}…)")
        import time as _time
        poll_start_time = _time.time()
        avg_video = get_avg_video_time(model_label)

        def on_poll(elapsed, status):
            # 평균 영상 생성 시간 기준으로 진행률 계산 (최대 0.99)
            frac = min(0.15 + (elapsed / avg_video) * 0.84, 0.99)
            progress(frac, desc=f"처리 중… {elapsed}초 경과 / 예상 {int(avg_video)}초 (상태: {status})")

        try:
            video_url = poll_task_result(
                access_key=ak, secret_key=sk,
                task_id=task_id, endpoint=endpoint,
                timeout=300, poll_interval=5,
                progress_callback=on_poll,
            )
            # 폴링 완료 후 실제 소요 시간 기록
            record_video_generation(model_label, _time.time() - poll_start_time)
        except TimeoutError as e:
            video_gallery_state.fill_pending_item(gallery_index, "", "", "failed", str(e))
            return None, f"⏱️ {e}"
        except Exception as e:
            video_gallery_state.fill_pending_item(gallery_index, "", "", "failed", str(e))
            return None, f"❌ 생성 실패: {e}"

        progress(0.97, desc="영상 다운로드 중...")
        try:
            video_bytes = download_video(video_url)
        except Exception as e:
            video_gallery_state.fill_pending_item(gallery_index, "", "", "failed", str(e))
            return None, f"❌ 영상 다운로드 실패: {e}"

        saved_video_path, thumb_path = save_video(video_bytes, model_label, prompt or "")
        video_gallery_state.fill_pending_item(
            gallery_index, saved_video_path, thumb_path, "success"
        )
        progress(1.0, desc="완료!")
        return saved_video_path, f"✅ 영상 생성 완료! 저장 위치: {saved_video_path}"

    return generate_video


def build_generate_fn(gallery_state: GalleryState):
    """생성 버튼 클릭 핸들러 팩토리 (즉시 반환 후 백그라운드 스레드로 병렬 생성)"""

    def generate(
        model_name: str,
        prompt: str,
        ratio: str,
        quality: str,
        count: int,
        ref_images,          # Gradio File 컴포넌트 값 (list of paths or None)
    ):
        import threading as _threading

        cfg = load_settings()
        api_key = cfg.get("api_key", "")
        if not api_key or not api_key.strip():
            gr.Warning("API 키 설정 탭에서 Google API 키를 먼저 저장해주세요.")
            return gallery_state.to_gradio_gallery(), gallery_state.get_summary()

        if not prompt or not prompt.strip():
            gr.Warning("프롬프트를 입력해주세요.")
            return gallery_state.to_gradio_gallery(), gallery_state.get_summary()

        count = max(1, min(int(count), MAX_COUNT))

        # 레퍼런스 이미지 로드
        ref_pil_images: list[Image.Image] = []
        ref_paths: list[str] = []
        if ref_images:
            raw_paths = [r if isinstance(r, str) else r.path for r in ref_images if r is not None]
            ref_pil_images = load_reference_images(raw_paths[:MAX_REFERENCE_IMAGES])
            # 새로고침 후에도 경로가 유효하도록 영구 저장소에 복사
            persistent_paths = _save_ref_images_persistent(raw_paths[:MAX_REFERENCE_IMAGES])
            ref_paths = persistent_paths if persistent_paths else raw_paths[:MAX_REFERENCE_IMAGES]

        # ① 즉시 N개의 대기 중 슬롯 생성
        allocated_indices = gallery_state.allocate_pending_items(
            count, model_name, ratio, quality, prompt
        )

        # ② 백그라운드 스레드에서 실제 이미지 생성 및 갤러리 상태 업데이트
        def run_generation():
            def on_progress(idx: int, img, err):
                gallery_index = allocated_indices[idx]
                if img is not None:
                    try:
                        img_path, _, thumb_path = save_image(
                            img, model_name, ratio, prompt, quality,
                            reference_image_paths=ref_paths,
                        )
                        gallery_state.fill_pending_item(gallery_index, img_path, thumb_path, "success", reference_image_paths=ref_paths)
                    except Exception as e:
                        gallery_state.fill_pending_item(gallery_index, "", "", "failed", str(e))
                else:
                    gallery_state.fill_pending_item(gallery_index, "", "", "failed", err)

            try:
                generate_batch_images(
                    api_key=api_key.strip(),
                    model_name=model_name,
                    prompt=prompt,
                    ratio=ratio,
                    quality=quality,
                    count=count,
                    reference_images=ref_pil_images if ref_pil_images else None,
                    progress_callback=on_progress,
                )
            except Exception as e:
                gallery_state.fail_remaining_pending(allocated_indices, repr(e))

        gen_thread = _threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        # ③ 즉시 반환 — gr.Timer 가 주기적으로 갤러리를 갱신함
        return gallery_state.to_gradio_gallery(), f"⏳ {count}장 생성 시작..."

    return generate


def build_download_zip_fn(gallery_state: GalleryState):
    def download_zip():
        paths = gallery_state.image_paths
        if not paths:
            gr.Warning("다운로드할 이미지가 없습니다.")
            return None
        zip_path = create_zip_from_paths(paths)
        return zip_path

    return download_zip


def build_download_single_fn(gallery_state: GalleryState):
    """선택된 이미지를 원본 PNG 파일로 다운로드하는 핸들러 팩토리"""

    def download_single(idx: int):
        item = gallery_state.get_success_item_by_visual_index(idx)
        if item is None:
            gr.Warning("이미지를 먼저 클릭하여 선택해주세요.")
            return None
        if not item.image_path or not os.path.exists(item.image_path):
            gr.Warning("이미지 파일을 찾을 수 없습니다.")
            return None
        return item.image_path

    return download_single


def build_clear_fn(gallery_state: GalleryState):
    def clear_gallery():
        gallery_state.clear()
        return [], "갤러리가 초기화되었습니다.", None

    return clear_gallery


def build_download_selected_fn(gallery_state: GalleryState):
    """다중 선택된 이미지를 ZIP으로 다운로드하는 핸들러 팩토리"""

    def download_selected(selected_json: str):
        try:
            indices = json.loads(selected_json or "[]")
        except (json.JSONDecodeError, ValueError):
            indices = []
        if not indices:
            gr.Warning("다운로드할 이미지를 먼저 선택해주세요.")
            return None
        paths = []
        for raw_idx in indices:
            try:
                item = gallery_state.get_success_item_by_visual_index(int(raw_idx))
                if item and item.image_path and os.path.exists(item.image_path):
                    paths.append(item.image_path)
            except (ValueError, TypeError):
                continue
        if not paths:
            gr.Warning("선택된 이미지 파일을 찾을 수 없습니다.")
            return None
        return create_zip_from_paths(paths)

    return download_selected


def build_smart_download_fn(gallery_state: GalleryState):
    """단일 선택 → PNG, 다중 선택 → ZIP 스마트 다운로드 핸들러 팩토리"""

    def smart_download(selected_json: str, current_idx: int):
        try:
            indices = json.loads(selected_json or "[]")
        except (json.JSONDecodeError, ValueError):
            indices = []

        if len(indices) > 1:
            # 다중 선택: ZIP 다운로드
            paths = []
            for raw_idx in indices:
                try:
                    item = gallery_state.get_success_item_by_visual_index(int(raw_idx))
                    if item and item.image_path and os.path.exists(item.image_path):
                        paths.append(item.image_path)
                except (ValueError, TypeError):
                    continue
            if not paths:
                gr.Warning("선택된 이미지 파일을 찾을 수 없습니다.")
                return None
            return create_zip_from_paths(paths)
        elif len(indices) == 1:
            # 단일 선택 (뱃지): PNG 다운로드
            try:
                item = gallery_state.get_success_item_by_visual_index(int(indices[0]))
            except (ValueError, TypeError):
                item = None
        else:
            # 뱃지 선택 없음: 마지막 클릭 이미지
            item = gallery_state.get_success_item_by_visual_index(current_idx)

        if item is None:
            gr.Warning("이미지를 먼저 클릭하여 선택해주세요.")
            return None
        if not item.image_path or not os.path.exists(item.image_path):
            gr.Warning("이미지 파일을 찾을 수 없습니다.")
            return None
        return item.image_path

    return smart_download


def build_delete_selected_fn(gallery_state: GalleryState):
    """선택된 이미지를 디스크에서 삭제하고 갤러리 상태를 갱신하는 핸들러 팩토리"""

    def delete_selected(selected_json: str, current_idx: int):
        try:
            indices = json.loads(selected_json or "[]")
        except (json.JSONDecodeError, ValueError):
            indices = []

        if not indices and current_idx >= 0:
            indices = [current_idx]

        if not indices:
            gr.Warning("삭제할 이미지를 먼저 선택해주세요.")
            return gallery_state.to_gradio_gallery(), gallery_state.get_summary(), "[]", -1

        removed_paths = gallery_state.remove_by_visual_indices(
            [int(i) for i in indices]
        )
        for path in removed_paths:
            if _is_safe_output_path(path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                    meta_path = path.replace(".png", ".json")
                    if os.path.exists(meta_path):
                        os.remove(meta_path)
                    thumb_path = get_thumbnail_path(path)
                    if os.path.exists(thumb_path):
                        os.remove(thumb_path)
                except Exception:
                    pass

        deleted_count = len(removed_paths)
        return (
            gallery_state.to_gradio_gallery(),
            gallery_state.get_summary() + f" (🗑️ {deleted_count}장 삭제됨)",
            "[]",
            -1,
        )

    return delete_selected


# ── UI 빌더 ──────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    # 로딩 플레이스홀더 이미지 준비 (없으면 생성)
    ensure_placeholder_image()

    gallery_state = GalleryState()
    saved = load_settings()
    saved_api_key = saved.get("api_key", "")
    saved_kling_access = saved.get("kling_access_key", "")
    saved_kling_secret = saved.get("kling_secret_key", "")

    # 저장된 UI 생성 설정 복원 (새로고침 후에도 유지)
    saved_image_model = saved.get("image_model", DEFAULT_MODEL)
    if saved_image_model not in MODELS:
        saved_image_model = DEFAULT_MODEL
    saved_image_ratio = saved.get("image_ratio", DEFAULT_RATIO)
    if saved_image_ratio not in ASPECT_RATIOS:
        saved_image_ratio = DEFAULT_RATIO
    saved_image_quality = saved.get("image_quality", DEFAULT_QUALITY)
    if saved_image_quality not in QUALITY_OPTIONS:
        saved_image_quality = DEFAULT_QUALITY
    try:
        saved_image_count = max(1, min(int(saved.get("image_count", DEFAULT_COUNT)), MAX_COUNT))
    except (TypeError, ValueError):
        saved_image_count = DEFAULT_COUNT
    saved_image_prompt = saved.get("image_prompt", "")

    saved_video_model = saved.get("video_model", KLING_DEFAULT_MODEL)
    if saved_video_model not in KLING_MODELS:
        saved_video_model = KLING_DEFAULT_MODEL
    saved_video_quality = saved.get("video_quality", KLING_DEFAULT_QUALITY)
    if saved_video_quality not in KLING_QUALITY_OPTIONS:
        saved_video_quality = KLING_DEFAULT_QUALITY
    try:
        saved_video_duration = max(3, min(int(saved.get("video_duration", KLING_DEFAULT_DURATION)), 15))
    except (TypeError, ValueError):
        saved_video_duration = KLING_DEFAULT_DURATION
    saved_video_ratio = saved.get("video_ratio", KLING_DEFAULT_RATIO)
    if saved_video_ratio not in KLING_VIDEO_RATIOS:
        saved_video_ratio = KLING_DEFAULT_RATIO
    saved_video_prompt = saved.get("video_prompt", "")
    saved_video_enable_audio = bool(saved.get("video_enable_audio", False))

    # 영구 저장된 레퍼런스 이미지 복원
    saved_ref_paths = _load_persistent_ref_paths()
    _initial_ref_preview_imgs = []
    for _rp in saved_ref_paths:
        try:
            with Image.open(str(_rp)) as _img:
                _img.draft('RGB', (_REF_THUMBNAIL_SIZE, _REF_THUMBNAIL_SIZE))
                _img.thumbnail((_REF_THUMBNAIL_SIZE, _REF_THUMBNAIL_SIZE), Image.LANCZOS)
                _initial_ref_preview_imgs.append(_img.convert("RGB").copy())
        except Exception:
            continue
    _initial_ref_cols = 1 if len(_initial_ref_preview_imgs) == 1 else 2

    # 기존 outputs/ 디렉토리의 이미지를 갤러리 상태에 로드
    for i, entry in enumerate(load_existing_outputs()):
        gallery_state.add(
            GalleryItem(
                image=None,
                image_path=entry["image_path"],
                thumbnail_path=entry.get("thumbnail_path", ""),
                model=entry["model"],
                ratio=entry["ratio"],
                quality=entry["quality"],
                prompt=entry["prompt"],
                index=i,
                status="success",
                reference_image_paths=entry.get("reference_image_paths", []),
            )
        )

    # 기존 outputs/ 디렉토리의 영상을 영상 갤러리 상태에 로드
    video_gallery_state = VideoGalleryState()
    for i, entry in enumerate(load_existing_video_outputs()):
        video_gallery_state.add(
            VideoGalleryItem(
                video_path=entry["path"],
                thumbnail_path=entry.get("thumbnail_path", PLACEHOLDER_IMAGE_PATH),
                model=entry["model"],
                prompt=entry["prompt"],
                ratio="",
                quality="",
                duration=0,
                index=i,
                status="success",
            )
        )

    def _use_as_ref(idx: int, current_files: list):
        """선택된 이미지를 레퍼런스 이미지 업로드 칸에 추가하는 공용 핸들러."""
        item = gallery_state.get_success_item_by_visual_index(idx)
        if item is None:
            gr.Warning("이미지를 먼저 클릭하여 선택해주세요.")
            return gr.update()

        current_paths = []
        if current_files:
            current_paths = [f if isinstance(f, str) else f.path for f in current_files if f is not None]

        if len(current_paths) >= MAX_REFERENCE_IMAGES:
            gr.Warning(f"레퍼런스 이미지는 최대 {MAX_REFERENCE_IMAGES}장까지 추가할 수 있습니다.")
            return gr.update()

        return gr.update(value=current_paths + [item.image_path])

    def _use_as_ref_selected(selected_json: str, current_files: list, current_idx: int = -1):
        """체크박스로 선택된 이미지들을 레퍼런스 이미지 업로드 칸에 추가."""
        try:
            indices = json.loads(selected_json or "[]")
        except (json.JSONDecodeError, ValueError):
            indices = []
        if not indices and current_idx >= 0:
            indices = [current_idx]
        if not indices:
            gr.Warning("레퍼런스로 추가할 이미지를 먼저 선택해주세요.")
            return gr.update()
        current_paths = []
        if current_files:
            current_paths = [f if isinstance(f, str) else f.path for f in current_files if f is not None]
        new_paths = []
        for raw_idx in indices:
            try:
                item = gallery_state.get_success_item_by_visual_index(int(raw_idx))
                if item and item.image_path and os.path.exists(item.image_path):
                    new_paths.append(item.image_path)
            except (ValueError, TypeError):
                continue
        if not new_paths:
            gr.Warning("선택된 이미지 파일을 찾을 수 없습니다.")
            return gr.update()
        combined = current_paths + new_paths
        if len(combined) > MAX_REFERENCE_IMAGES:
            gr.Warning(f"레퍼런스 이미지는 최대 {MAX_REFERENCE_IMAGES}장까지 추가할 수 있습니다.")
            combined = combined[:MAX_REFERENCE_IMAGES]
        return gr.update(value=combined)

    def _use_as_ref_single(val: str, current_files: list):
        """오버레이 버튼에서 단일 이미지를 레퍼런스로 추가 ('idx:timestamp' 값 처리)."""
        try:
            idx = int(val.split(":")[0])
        except Exception:
            idx = -1
        return _use_as_ref(idx, current_files)

    # 탭 전환을 위한 JavaScript (localStorage로 새로고침 시 탭 유지)
    TAB_PERSIST_JS = """
    <script>
    // ── 탭 지속성 ────────────────────────────────────────────────────────────
    (function() {
        var TAB_KEY = 'keyImageGenTab';
        function saveTab(idx) { try { localStorage.setItem(TAB_KEY, String(idx)); } catch(e) { console.warn('Failed to save tab:', e); } }
        function getTabBtns() {
            var r = document.querySelectorAll('[role="tab"]');
            return r.length ? Array.from(r) : Array.from(document.querySelectorAll('.tab-nav button'));
        }
        function restoreTab() {
            try {
                var idx = localStorage.getItem(TAB_KEY);
                if (idx === null) return;
                var attempt = 0;
                var iv = setInterval(function() {
                    var btns = getTabBtns();
                    if (btns.length > parseInt(idx)) { btns[parseInt(idx)].click(); clearInterval(iv); }
                    if (++attempt > 60) clearInterval(iv);
                }, 100);
            } catch(e) {}
        }
        document.addEventListener('click', function(e) {
            var btn = e.target.closest('[role="tab"]') || e.target.closest('.tab-nav button');
            if (btn) { var btns = getTabBtns(); var i = btns.indexOf(btn); if (i >= 0) saveTab(i); }
        }, true);
        document.readyState === 'loading'
            ? document.addEventListener('DOMContentLoaded', function() { setTimeout(restoreTab, 500); })
            : setTimeout(restoreTab, 500);
    })();

    // ── 공통 유틸리티: 썸네일 URL → 원본 URL 변환 ──────────────────────────
    function thumbSrcToOriginalSrc(src) {
        // thumbs/ 서브폴더 경로를 원본 경로로 변환
        // 예: /gradio_api/file=/path/outputs/2026-04-09/thumbs/file.png
        //  → /gradio_api/file=/path/outputs/2026-04-09/file.png
        return src.replace(new RegExp('/thumbs/([^/?]+)([?]|$)'), '/$1$2');
    }

    // ── 공통 헬퍼: Gradio/Svelte 감지 가능한 네이티브 값 설정 ────────────────
    function setNativeValue(inp, val) {
        var proto = inp.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
        var nativeSetter = Object.getOwnPropertyDescriptor(proto, 'value');
        if (nativeSetter && nativeSetter.set) { nativeSetter.set.call(inp, val); } else { inp.value = val; }
        inp.dispatchEvent(new InputEvent('input', {bubbles: true, cancelable: true}));
        inp.dispatchEvent(new Event('change', {bubbles: true}));
    }

    // ── 공통 헬퍼: 갤러리 썸네일 항목 취득 (Gradio 버전별 선택자 자동 탐색) ──
    // Gradio 5.x 버전에 따라 data-testid, 클래스명이 달라질 수 있으므로 다중 선택자를 순차 시도
    var _THUMB_SELS = [
        '.gallery-item',
        '[data-testid^="thumbnail"]',
        '[data-testid="thumbnail"]',
        '.thumbnail-item',
        '[class*="thumbnail-item"]',
        'button[class*="thumbnail"]',
        '[data-testid="gallery-item"]'
    ];
    // 갤러리 컨테이너 내 모든 썸네일 요소 배열 반환
    window.__getGalleryItems = function(containerId) {
        var g = document.getElementById(containerId);
        if (!g) return [];
        for (var i = 0; i < _THUMB_SELS.length; i++) {
            try {
                var r = Array.from(g.querySelectorAll(_THUMB_SELS[i]))
                    .filter(function(el) { return !el.closest('.preview'); });
                if (r.length) return r;
            } catch(e) {}
        }
        // 최종 폴백: gallery-container 직접 자식 div 또는 img를 포함하는 button
        try {
            var gc = g.querySelector('.gallery-container, [class*="gallery-container"]');
            if (gc) {
                var kids = Array.from(gc.children).filter(function(c) {
                    return c.querySelector('img') && !c.classList.contains('preview');
                });
                if (kids.length) return kids;
            }
        } catch(e) {}
        try {
            var btns = Array.from(g.querySelectorAll('button'));
            var imgBtns = btns.filter(function(b) { return b.querySelector('img') && !b.closest('.preview'); });
            if (imgBtns.length) return imgBtns;
        } catch(e) {}
        return [];
    };
    // 주어진 요소의 조상 중 썸네일 요소 반환 (closest 방식)
    window.__getGalleryItem = function(el) {
        if (el.closest && el.closest('.preview')) return null;  // 라이트박스 내부 요소 제외
        for (var i = 0; i < _THUMB_SELS.length; i++) {
            try {
                var r = el.closest(_THUMB_SELS[i]);
                if (r) return r;
            } catch(e) {}
        }
        // 최종 폴백: gallery-container 직접 자식 중 img를 포함하는 요소
        try {
            var gc = el.closest('.gallery-container, [class*="gallery-container"]');
            if (gc) {
                var cur = el;
                while (cur && cur.parentElement !== gc) cur = cur.parentElement;
                if (cur && cur.parentElement === gc && cur.querySelector('img')) return cur;
            }
        } catch(e) {}
        return null;
    };

    // ── 갤러리 호버 플로팅 오버레이 ─────────────────────────────────────────
    (function() {
        var CFGS = [
            {id:'live-gallery', dl:'btn-download-gen',     vid:'btn-video-gen',     ref:'btn-ref-gen',     btns:['dl','vid','ref']},
            {id:'full-gallery', dl:'btn-download-gallery', vid:'btn-video-gallery', ref:'btn-ref-gallery', btns:['dl','vid','ref']},
            {id:'video-gallery', dl:'btn-download-video', btns:['dl']}
        ];

        // body 최상위에 단일 플로팅 오버레이 생성 (갤러리 아이템 DOM 수정 없음)
        var ov = document.createElement('div');
        ov.id = 'gha-ov';
        ov.style.cssText = [
            'position:fixed', 'display:none', 'align-items:flex-end', 'justify-content:flex-end',
            'padding:6px', 'gap:5px',
            'z-index:10000',
            'box-sizing:border-box',
            'pointer-events:none'
        ].join(';');

        [{key:'dl',e:'⬇️',t:'PNG 다운로드',l:'PNG'},{key:'vid',e:'🎬',t:'영상화',l:'영상화'},{key:'ref',e:'🖼️',t:'레퍼런스',l:'레퍼런스'}]
        .forEach(function(b) {
            var el = document.createElement('button');
            el.dataset.k = b.key; el.title = b.t;
            el.innerHTML = b.e + ' <span style="font-size:0.78rem">' + b.l + '</span>';
            el.style.cssText = [
                'background:rgba(20,20,20,0.72)', 'border:none', 'border-radius:5px',
                'font-size:0.9rem', 'cursor:pointer', 'padding:3px 8px', 'height:28px',
                'display:flex', 'align-items:center', 'justify-content:center', 'gap:3px',
                'color:#fff', 'white-space:nowrap', 'font-weight:500',
                'box-shadow:0 1px 4px rgba(0,0,0,0.5)', 'flex-shrink:0', 'pointer-events:auto',
                'backdrop-filter:blur(2px)'
            ].join(';');
            ov.appendChild(el);
        });
        document.body.appendChild(ov);

        var curItem = null, curCfg = null, curItemIdx = -1;
        window.__ovIdx = -1;  // 오버레이 호버 인덱스 (버튼 클릭 시 주 버튼에 전달)

        function cfgOf(el) {
            for (var i = 0; i < CFGS.length; i++) {
                var g = document.getElementById(CFGS[i].id);
                if (g && g.contains(el)) return CFGS[i];
            }
            return null;
        }
        function itemOf(el) { return window.__getGalleryItem(el); }
        function getItems(cfg) { return window.__getGalleryItems(cfg.id); }
        function posOv(item) {
            var r = item.getBoundingClientRect();
            ov.style.left = r.left + 'px'; ov.style.top = r.top + 'px';
            ov.style.width = r.width + 'px'; ov.style.height = r.height + 'px';
            ov.style.display = 'flex';
            // 각 버튼에 현재 갤러리 ID·아이템 인덱스를 저장 (mousedown 시 curItem이 null이어도 동작)
            // posOv는 curCfg/curItemIdx 설정 직후 호출되므로 항상 유효한 값이 존재
            var visibleBtns = curCfg.btns || ['dl','vid','ref'];
            var btns = ov.querySelectorAll('[data-k]');
            for (var bi = 0; bi < btns.length; bi++) {
                btns[bi].dataset.gid = curCfg.id;
                btns[bi].dataset.itemIdx = String(curItemIdx);
                btns[bi].style.display = visibleBtns.indexOf(btns[bi].dataset.k) >= 0 ? 'flex' : 'none';
            }
        }
        function hideOv() {
            var prevGid = curCfg ? curCfg.id : null;
            ov.style.display = 'none'; curItem = null; curCfg = null; curItemIdx = -1;
            window.__ovIdx = -1;
            if (prevGid !== null && window.__msUpdateHover) window.__msUpdateHover(prevGid, -1);
        }
        function gClick(id) {
            var el = document.getElementById(id); if (!el) return;
            var b = el.tagName === 'BUTTON' ? el : el.querySelector('button'); if (b) b.click();
        }

        document.addEventListener('mouseover', function(e) {
            if (ov.contains(e.target)) return;
            var cfg = cfgOf(e.target); if (!cfg) { hideOv(); return; }
            var item = itemOf(e.target); if (!item) { hideOv(); return; }
            var newIdx = getItems(cfg).indexOf(item);
            var changed = (curCfg !== cfg || curItemIdx !== newIdx);
            curItem = item; curCfg = cfg; curItemIdx = newIdx;
            posOv(item);
            if (changed && window.__msUpdateHover) window.__msUpdateHover(cfg.id, newIdx);
        });
        ov.addEventListener('mouseleave', hideOv);

        // 갤러리 img src에서 직접 파일 다운로드 (썸네일 → 원본 경로로 변환 후 다운로드)
        function downloadFromGallery(gid, idx) {
            var items = window.__getGalleryItems(gid);
            var item = items[idx]; if (!item) return false;
            var img = item.querySelector('img');
            if (!img || !img.src || img.src.indexOf('data:') === 0) return false;
            // 썸네일 src를 원본 src로 변환
            var origSrc = thumbSrcToOriginalSrc(img.src);
            var rawSrc = decodeURIComponent(origSrc);
            var filename = rawSrc.split('=').pop().split('/').pop().split('?')[0];
            if (!filename || !filename.match(new RegExp('[.](png|jpg|jpeg|webp|gif)$', 'i'))) filename = 'image.png';
            var a = document.createElement('a');
            a.href = origSrc;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            return true;
        }
        // 오버레이 버튼 클릭: 라이트박스 열지 않고 서버 액션 직접 트리거
        function triggerOverlayAction(gid, key, idx) {
            var tbMap = {
                'live-gallery': {dl:'overlay-dl-gen', vid:'overlay-vid-gen', ref:'overlay-ref-gen'},
                'full-gallery': {dl:'overlay-dl-gallery', vid:'overlay-vid-gallery', ref:'overlay-ref-gallery'},
                'video-gallery': {dl:'overlay-dl-video'}
            };
            var tbId = tbMap[gid] && tbMap[gid][key]; if (!tbId) return;
            var wrapper = document.getElementById(tbId); if (!wrapper) return;
            var inp = wrapper.querySelector('input[type="text"],textarea'); if (!inp) return;
            // idx:timestamp 형식으로 설정하여 같은 인덱스 반복 클릭도 이벤트 발생
            setNativeValue(inp, idx + ':' + Date.now());
        }
        ov.addEventListener('mousedown', function(e) {
            var btn = e.target.closest('[data-k]');
            if (!btn) return;
            e.preventDefault(); e.stopPropagation();
            // 버튼에 저장된 gid/itemIdx 우선 사용 (hover 상태가 해제된 후에도 안정적으로 동작)
            var gid = btn.dataset.gid || (curCfg ? curCfg.id : '');
            var idx = parseInt(btn.dataset.itemIdx, 10);
            if (isNaN(idx) || idx < 0) idx = curItemIdx;
            if (!gid || idx < 0) return;
            // window.__ovIdx 를 설정하고 해당 주 버튼을 클릭 →
            // 각 버튼의 js 핸들러가 __ovIdx 를 감지하여 올바른 인덱스로 동작
            window.__ovIdx = idx;
            for (var ci = 0; ci < CFGS.length; ci++) {
                if (CFGS[ci].id === gid) {
                    var btnId = CFGS[ci][btn.dataset.k];
                    if (btnId) gClick(btnId);
                    break;
                }
            }
        });
        window.addEventListener('scroll', function() { if (curItem && ov.style.display !== 'none') posOv(curItem); }, true);
        window.addEventListener('resize', function() { if (curItem && ov.style.display !== 'none') posOv(curItem); });
    })();

    // ── 다중 선택 (호버 시 체크박스 표시, 선택 시 항상 표시) ────────────────
    (function() {
        var GIDS = ['live-gallery', 'full-gallery', 'video-gallery'];
        var TBIDS = {'live-gallery': 'ms-state-gen', 'full-gallery': 'ms-state-gallery', 'video-gallery': 'ms-state-video'};
        var sels = {}, containers = {};
        var hoverGid = null, hoverIdx = -1;
        GIDS.forEach(function(id) { sels[id] = new Set(); containers[id] = null; });

        window.__msUpdateHover = function(gid, idx) {
            var prevGid = hoverGid;
            hoverGid = (idx >= 0) ? gid : null;
            hoverIdx = idx;
            if (prevGid) updateBadgeOpacity(prevGid);
            if (gid) updateBadgeOpacity(gid);
        };
        window.__msGetSels = function(gid) { return sels[gid] ? [...sels[gid]] : []; };
        window.__msClear = function(gid) {
            if (sels[gid]) { sels[gid].clear(); syncTextbox(gid); refreshOverlays(gid); }
        };
        window.__msToggle = function(gid, idx) {
            if (!sels[gid]) return;
            if (sels[gid].has(idx)) sels[gid].delete(idx); else sels[gid].add(idx);
            refreshOverlays(gid); syncTextbox(gid);
        };
        window.__msToggleAll = function(gid) {
            if (!sels[gid]) return;
            if (sels[gid].size > 0) {
                sels[gid].clear();
            } else {
                var items = getItems(gid);
                for (var i = 0; i < items.length; i++) { sels[gid].add(i); }
            }
            refreshOverlays(gid); syncTextbox(gid);
        };

        function updateBadgeOpacity(gid) {
            var c = containers[gid]; if (!c) return;
            var isHoverGid = (hoverGid === gid);
            for (var i = 0; i < c.children.length; i++) {
                var badge = c.children[i].querySelector('.ms-badge'); if (!badge) continue;
                var isSelected = sels[gid].has(i);
                var isHovered = isHoverGid && (i === hoverIdx);
                badge.style.opacity = (isSelected || isHovered) ? '1' : '0';
            }
        }
        function getItems(gid) { return window.__getGalleryItems(gid); }
        function getContainer(gid) {
            if (!containers[gid]) {
                var c = document.createElement('div');
                c.style.cssText = 'position:fixed;top:0;left:0;pointer-events:none;z-index:10001;';
                document.body.appendChild(c);
                containers[gid] = c;
            }
            return containers[gid];
        }
        function syncTextbox(gid) {
            var wrapper = document.getElementById(TBIDS[gid]); if (!wrapper) return;
            var inp = wrapper.querySelector('input[type="text"],textarea'); if (!inp) return;
            setNativeValue(inp, JSON.stringify([...sels[gid]]));
        }
        function refreshOverlays(gid) {
            var c = getContainer(gid), items = getItems(gid), sel = sels[gid];
            var isHoverGid = (hoverGid === gid);
            while (c.children.length > items.length) c.removeChild(c.lastChild);
            while (c.children.length < items.length) {
                var d = document.createElement('div');
                d.style.cssText = 'position:fixed;box-sizing:border-box;display:flex;align-items:flex-start;justify-content:flex-start;padding:4px;pointer-events:none;';
                var badge = document.createElement('div');
                badge.className = 'ms-badge';
                badge.style.cssText = [
                    'width:22px','height:22px','border-radius:50%',
                    'display:flex','align-items:center','justify-content:center',
                    'font-size:0.85rem','font-weight:700',
                    'box-shadow:0 1px 4px rgba(0,0,0,0.5)','user-select:none',
                    'pointer-events:auto','cursor:pointer',
                    'transition:opacity 0.15s, background 0.15s',
                    'border:2px solid rgba(255,255,255,0.7)','opacity:0'
                ].join(';');
                d.appendChild(badge);
                c.appendChild(d);
            }
            items.forEach(function(item, idx) {
                var r = item.getBoundingClientRect(), d = c.children[idx];
                d.style.left = r.left+'px'; d.style.top = r.top+'px';
                d.style.width = r.width+'px'; d.style.height = r.height+'px';
                var badge = d.querySelector('.ms-badge');
                var isSelected = sel.has(idx), isHovered = isHoverGid && (idx === hoverIdx);
                badge.style.background = isSelected ? '#4f46e5' : 'rgba(255,255,255,0.85)';
                badge.style.borderColor = isSelected ? '#4f46e5' : 'rgba(0,0,0,0.35)';
                badge.style.color = isSelected ? '#fff' : 'transparent';
                badge.textContent = isSelected ? '✓' : '';
                badge.style.opacity = (isSelected || isHovered) ? '1' : '0';
                badge.onclick = null;
                badge.onclick = (function(i) {
                    return function(e) {
                        e.stopPropagation(); e.preventDefault();
                        if (sel.has(i)) sel.delete(i); else sel.add(i);
                        refreshOverlays(gid); syncTextbox(gid);
                    };
                })(idx);
            });
        }
        var obs = new MutationObserver(function() { GIDS.forEach(refreshOverlays); });
        var _attached = {};
        function attachObs() {
            // 갤러리 요소가 아직 DOM에 없으면 500ms 후 재시도 (탭 지연 렌더링 대응)
            var allDone = true;
            GIDS.forEach(function(gid) {
                if (_attached[gid]) return;
                var g = document.getElementById(gid);
                if (g) { obs.observe(g, {childList: true, subtree: true}); _attached[gid] = true; }
                else allDone = false;
            });
            GIDS.forEach(refreshOverlays);
            if (!allDone) setTimeout(attachObs, 500);
        }
        window.addEventListener('scroll', function() { GIDS.forEach(refreshOverlays); }, true);
        window.addEventListener('resize', function() { GIDS.forEach(refreshOverlays); });
        setTimeout(attachObs, 300);
    })();

    // ── 다중 선택 모드에서 썸네일 클릭 시 라이트박스 차단 → 체크박스 토글 ────
    (function() {
        var CFGS = [
            {id:'live-gallery'}, {id:'full-gallery'}, {id:'video-gallery'}
        ];
        function cfgOf(el) {
            for (var i = 0; i < CFGS.length; i++) {
                var g = document.getElementById(CFGS[i].id);
                if (g && g.contains(el)) return CFGS[i];
            }
            return null;
        }
        function itemOf(el) { return window.__getGalleryItem(el); }
        function getItems(gid) { return window.__getGalleryItems(gid); }
        document.addEventListener('click', function(e) {
            var ov = document.getElementById('gha-ov');
            if (ov && ov.contains(e.target)) return;
            var cfg = cfgOf(e.target); if (!cfg) return;
            var gid = cfg.id;
            var sels = window.__msGetSels ? window.__msGetSels(gid) : [];
            if (!sels || !sels.length) return;
            var item = itemOf(e.target); if (!item) return;
            var items = getItems(gid);
            var idx = items.indexOf(item); if (idx < 0) return;
            e.preventDefault(); e.stopImmediatePropagation();
            if (window.__msToggle) window.__msToggle(gid, idx);
        }, true);
    })();

    // ── 라이트박스: 이미지 외부 영역 클릭 시 자동 닫기 ──────────────────────
    (function() {
        document.addEventListener('click', function(e) {
            // Gradio 6+: 라이트박스 이미지는 data-testid="detailed-image"
            // Gradio 5 이하: [data-testid="lightbox"] 또는 .lightbox
            var detailedImg = document.querySelector('[data-testid="detailed-image"]');
            var lb = detailedImg
                ? (detailedImg.closest('.preview') || detailedImg.parentElement)
                : (document.querySelector('[data-testid="lightbox"]') || document.querySelector('.lightbox'));
            if (!lb) return;

            var closeBtn = lb.querySelector('button[aria-label="Close"]')
                        || lb.querySelector('button[title="Close"]')
                        || Array.from(lb.querySelectorAll('button')).pop();
            if (!closeBtn) return;

            if (lb.contains(e.target)) {
                // 라이트박스 내부 클릭: 이미지·버튼·캡션이 아닌 배경 영역이면 닫기
                if (!e.target.closest('img')
                    && !e.target.closest('button')
                    && !e.target.closest('[data-testid="caption"]')
                    && !e.target.closest('.caption')
                    && !e.target.closest('[class*="caption"]')) {
                    closeBtn.click();
                }
            } else {
                // 라이트박스 외부 클릭: 항상 닫기 (툴바 버튼 등 포함)
                closeBtn.click();
            }
        }, true);
    })();

    // ── 라이트박스 열릴 때 썸네일 → 원본 이미지로 교체 ──────────────────────
    (function() {
        new MutationObserver(function(mutations) {
            mutations.forEach(function(m) {
                m.addedNodes.forEach(function(node) {
                    if (node.nodeType !== 1) return;
                    // Gradio 라이트박스는 fixed div 또는 .preview 클래스 div에 img를 포함
                    var imgs = [];
                    if (node.tagName === 'IMG') {
                        imgs = [node];
                    } else if (typeof node.querySelectorAll === 'function') {
                        // Gradio 6+: .preview 컨테이너 또는 data-testid="detailed-image"
                        var isGr6Preview = node.classList && node.classList.contains('preview');
                        var hasDetailedImg = !!node.querySelector('[data-testid="detailed-image"]');
                        // Gradio 5: 고정 위치 요소 또는 [data-testid="lightbox"]
                        var fixed = node.style && node.style.position === 'fixed';
                        if (isGr6Preview || hasDetailedImg || fixed || (node.closest && node.closest('[data-testid="lightbox"]'))) {
                            imgs = Array.from(node.querySelectorAll('img'));
                        }
                        // data-testid="lightbox" 내부 이미지 (Gradio 5)
                        if (!imgs.length) {
                            var lb = node.querySelector('[data-testid="lightbox"]');
                            if (lb) imgs = Array.from(lb.querySelectorAll('img'));
                        }
                    }
                    imgs.forEach(function(img) {
                        if (!img.src || img.src.indexOf('/thumbs/') === -1) return;
                        var origSrc = thumbSrcToOriginalSrc(img.src);
                        if (origSrc !== img.src) img.src = origSrc;
                    });
                });
            });
        }).observe(document.body, {childList: true, subtree: true});
    })();

    // ── 라이트박스 열릴 때 레퍼런스 이미지 패널 표시 ─────────────────────────
    (function() {
        var PANEL_ID = 'lb-ref-panel';
        var pollTimer = null;
        var pollCount = 0;
        var MAX_POLLS = 17; // 300ms * 17 ≈ 5s
        var lbOpen = false;
        var pollInitialSrcs = null;

        function getLightbox() {
            // Gradio 6+: 라이트박스 이미지는 data-testid="detailed-image"
            var detailedImg = document.querySelector('[data-testid="detailed-image"]');
            if (detailedImg) {
                return detailedImg.closest('.preview') || detailedImg.parentElement;
            }
            return document.querySelector('[data-testid="lightbox"]')
                || document.querySelector('.lightbox');
        }

        function getRefImgUrls() {
            // lb-ref-urls-state 텍스트박스에서 절대 경로 JSON을 읽어 Gradio 파일 URL 생성
            var el = document.getElementById('lb-ref-urls-state');
            if (el) {
                var inp = el.querySelector('textarea, input[type="text"]');
                if (inp && inp.value) {
                    try {
                        var paths = JSON.parse(inp.value);
                        if (Array.isArray(paths) && paths.length) {
                            return paths.map(function(p) { return '/gradio_api/file=' + p; });
                        }
                    } catch(e) {}
                }
            }
            return [];
        }

        function stopPolling() {
            if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
            pollCount = 0;
        }

        function removePanel() {
            var p = document.getElementById(PANEL_ID);
            if (p) p.remove();
        }

        function renderPanel(lb, refUrls) {
            var panel = document.getElementById(PANEL_ID);
            if (!panel) {
                panel = document.createElement('div');
                panel.id = PANEL_ID;
                panel.style.cssText = [
                    'position:fixed', 'bottom:16px', 'left:50%',
                    'transform:translateX(-50%)',
                    'z-index:20000',
                    'background:rgba(10,10,20,0.85)',
                    'border-radius:10px',
                    'padding:8px 14px',
                    'display:flex', 'flex-direction:column', 'align-items:center', 'gap:6px',
                    'max-width:90vw',
                    'box-shadow:0 4px 20px rgba(0,0,0,0.6)',
                    'pointer-events:none',
                    'backdrop-filter:blur(6px)'
                ].join(';');
                var title = document.createElement('div');
                title.style.cssText = 'color:rgba(255,255,255,0.7);font-size:0.72rem;font-weight:600;letter-spacing:0.04em;';
                title.textContent = '📎 사용된 레퍼런스 이미지';
                panel.appendChild(title);
                var row = document.createElement('div');
                row.id = PANEL_ID + '-row';
                row.style.cssText = 'display:flex;flex-direction:row;gap:8px;flex-wrap:wrap;justify-content:center;max-width:80vw;';
                panel.appendChild(row);
                // 라이트박스 내부에 삽입 → 라이트박스 닫힐 때 패널도 자동 제거
                lb.appendChild(panel);
            }

            var row = document.getElementById(PANEL_ID + '-row');
            if (!row) return;
            var newSrcs = refUrls.join(',');
            if (row.dataset.srcs === newSrcs) return;
            row.dataset.srcs = newSrcs;
            row.innerHTML = '';
            refUrls.forEach(function(src) {
                var thumb = document.createElement('img');
                thumb.src = src;
                // 레퍼런스 미리보기 썸네일과 동일한 크기 (256px 기준)
                thumb.style.cssText = 'height:64px;max-width:96px;width:auto;object-fit:contain;border-radius:5px;background:rgba(255,255,255,0.06);';
                row.appendChild(thumb);
            });
        }

        function pollForUrls() {
            var lb = getLightbox();
            if (!lb) { stopPolling(); lbOpen = false; return; }
            var refUrls = getRefImgUrls();
            var currentSrcs = JSON.stringify(refUrls);
            // Python 핸들러가 완료되어 텍스트박스 값이 바뀐 경우에만 렌더링
            if (currentSrcs !== pollInitialSrcs) {
                if (refUrls.length) renderPanel(lb, refUrls);
                else removePanel();
                stopPolling();
                return;
            }
            pollCount++;
            if (pollCount >= MAX_POLLS) {
                // 5초 타임아웃: 값이 바뀌지 않았으면 현재 값으로 폴백 표시
                if (refUrls.length) renderPanel(lb, refUrls);
                stopPolling();
            }
        }

        function startPolling() {
            stopPolling();
            pollCount = 0;
            removePanel();
            // 폴링 시작 시점의 텍스트박스 값을 기록 (Python 핸들러 반환 감지용)
            pollInitialSrcs = JSON.stringify(getRefImgUrls());
            pollTimer = setInterval(pollForUrls, 300);
        }

        // 라이트박스 열림/닫힘을 단일 Observer로 감지
        new MutationObserver(function() {
            var lb = getLightbox();
            if (lb && !lbOpen) {
                lbOpen = true;
                startPolling();
            } else if (!lb && lbOpen) {
                lbOpen = false;
                stopPolling();
            }
        }).observe(document.body, {childList: true, subtree: true});
    })();

    // ── 라이트박스 캡션 프롬프트 복사 버튼 ──────────────────────────────────
    (function() {
        var BTN_ID = 'lb-copy-prompt-btn';
        var copyTimer = null;

        function getCaptionEl() {
            // Gradio 6+: <caption class="caption svelte-..."> inside .preview
            var preview = document.querySelector('.preview');
            if (preview) {
                var cap = preview.querySelector('caption.caption, [class*="caption"]');
                if (cap && cap.textContent.trim()) return cap;
            }
            // Gradio 5 fallback
            var lb = document.querySelector('[data-testid="lightbox"]') || document.querySelector('.lightbox');
            if (lb) return lb.querySelector('[data-testid="caption"], .caption, [class*="caption"]');
            return null;
        }

        function extractPrompt(fullText) {
            // 캡션 형식: "#N | model | ratio | quality\\nprompt text"
            // 첫 번째 줄바꿈 이후가 프롬프트 텍스트
            var nl = fullText.indexOf('\\n');
            return (nl >= 0 ? fullText.slice(nl + 1) : fullText).trim();
        }

        function removeBtn() {
            var b = document.getElementById(BTN_ID);
            if (b) b.remove();
            if (copyTimer) { clearTimeout(copyTimer); copyTimer = null; }
        }

        function updateBtn() {
            var cap = getCaptionEl();
            if (!cap) { removeBtn(); return; }

            var prompt = extractPrompt(cap.textContent || '');
            if (!prompt) { removeBtn(); return; }

            var btn = document.getElementById(BTN_ID);
            if (!btn) {
                btn = document.createElement('button');
                btn.id = BTN_ID;
                btn.title = '프롬프트 복사';
                btn.textContent = '📋 복사';

                function showCopySuccess() {
                    btn.textContent = '✅ 복사됨';
                    if (copyTimer) clearTimeout(copyTimer);
                    copyTimer = setTimeout(function() { btn.textContent = '📋 복사'; }, 1500);
                }
                function showCopyError() {
                    btn.textContent = '❌ 실패';
                    if (copyTimer) clearTimeout(copyTimer);
                    copyTimer = setTimeout(function() { btn.textContent = '📋 복사'; }, 1500);
                }

                btn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    var text = extractPrompt((getCaptionEl() || {textContent:''}).textContent || '');
                    if (!text) return;
                    navigator.clipboard.writeText(text).then(showCopySuccess).catch(function() {
                        // clipboard API 실패 시 execCommand 폴백 (deprecated이지만 구형 브라우저 대응)
                        try {
                            var ta = document.createElement('textarea');
                            ta.value = text;
                            ta.style.cssText = 'position:fixed;left:-9999px;top:0';
                            document.body.appendChild(ta);
                            ta.select();
                            var ok = document.execCommand('copy');
                            document.body.removeChild(ta);
                            if (ok) { showCopySuccess(); } else { showCopyError(); }
                        } catch (_) {
                            showCopyError();
                        }
                    });
                });
                document.body.appendChild(btn);
            }

            // 캡션 요소 위치를 기반으로 버튼 위치 지정 (오른쪽 상단)
            var rect = cap.getBoundingClientRect();
            btn.style.top = (rect.top + 4) + 'px';
            btn.style.right = (window.innerWidth - rect.right + 4) + 'px';
            btn.style.left = 'auto';
        }

        new MutationObserver(function() {
            updateBtn();
        }).observe(document.body, {childList: true, subtree: true, characterData: true});

        // 스크롤/리사이즈 시 버튼 위치 재계산
        window.addEventListener('scroll', updateBtn, true);
        window.addEventListener('resize', updateBtn);
    })();

    // ── Ctrl+Enter 단축키: 이미지/영상 생성 ──────────────────────────────
    (function() {
        function bindCtrlEnter(promptElemId, btnElemId) {
            var attempts = 0;
            (function tryBind() {
                var wrap = document.getElementById(promptElemId);
                var ta = wrap ? wrap.querySelector('textarea') : null;
                if (!ta) {
                    if (++attempts < 40) { setTimeout(tryBind, 300); return; }
                    return;
                }
                ta.addEventListener('keydown', function(e) {
                    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                        e.preventDefault(); e.stopPropagation();
                        var bw = document.getElementById(btnElemId);
                        var b = bw ? (bw.tagName === 'BUTTON' ? bw : bw.querySelector('button')) : null;
                        if (b) b.click();
                    }
                });
            })();
        }
        bindCtrlEnter('image-prompt', 'image-generate-btn');
        bindCtrlEnter('video-prompt', 'video-generate-btn');
    })();

    // ── 다운로드 파일 위젯 자동 실행 (별도 창 없이 바로 다운로드) ──────────
    (function() {
        var DL_IDS = ['single-png-gen', 'selected-zip-gen', 'single-png-gallery', 'selected-zip-gallery', 'full-zip-gallery', 'single-video-gen', 'selected-videos-zip-gen'];
        DL_IDS.forEach(function(id) {
            var attempts = 0;
            (function trySetup() {
                var wrap = document.getElementById(id);
                if (!wrap) {
                    if (++attempts < 40) { setTimeout(trySetup, 500); return; }
                    return;
                }
                // 위젯을 시각적으로 숨김
                wrap.style.display = 'none';
                // 파일 링크 생성 시 자동으로 다운로드 트리거 (_lastDlHref 비교로 중복 방지)
                var _lastDlHref = '';
                new MutationObserver(function() {
                    var link = wrap.querySelector('a[href]');
                    if (!link || !link.href || link.href === _lastDlHref) return;
                    _lastDlHref = link.href;
                    var a = document.createElement('a');
                    a.href = link.href;
                    // download 속성 우선, 없으면 URL에서 파일명 추출
                    var fname = link.getAttribute('download') || link.href.split('/').pop().split('?')[0] || '';
                    a.download = fname;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    setTimeout(function() { _lastDlHref = ''; }, 5000);
                }).observe(wrap, {childList: true, subtree: true, attributes: true});
            })();
        });
    })();
    // ── js 파라미터 공용 헬퍼 함수 ────────────────────────────────────────────
    window.__getOvIdx = function(idx) {
        var oi = (typeof window.__ovIdx !== 'undefined' && window.__ovIdx >= 0) ? window.__ovIdx : idx;
        window.__ovIdx = -1;
        return oi;
    };
    window.__getSelJson = function(gid, ms) {
        var sels = window.__msGetSels ? window.__msGetSels(gid) : [];
        return JSON.stringify(sels.length ? sels : JSON.parse(ms || '[]'));
    };
    </script>
    """

    with gr.Blocks(
        title="🎨 키 이미지 생성 툴",
    ) as demo:

        gr.HTML(
            """
            <div class="title-text">🎨 AI 영상 키 이미지 생성 툴</div>
            <div class="subtitle-text">나노 바나나 2 / 나노 바나나 프로 모델로 영상 키 이미지를 생성하고, Kling AI로 영상을 만들어보세요.</div>
            """
        )
        gr.HTML(value="", head=TAB_PERSIST_JS)

        # 탭 간 이미지 전달용 상태
        selected_img_idx_gen = gr.State(-1)      # 이미지 생성 탭 갤러리 선택 인덱스
        selected_img_idx_gallery = gr.State(-1)  # 갤러리 탭 선택 인덱스

        with gr.Tabs() as main_tabs:

            # ── 탭 1: API 키 설정 ────────────────────────────────────────────
            with gr.Tab("🔑 API 키 설정", id="tab_keys"):
                gr.Markdown(
                    """
                    ### Google Gemini API 키 설정
                    [Google AI Studio](https://aistudio.google.com/app/apikey)에서 API 키를 발급받으세요.
                    입력한 키는 로컬 파일(`.app_settings.json`)에 저장됩니다.
                    """
                )
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="Google API 키",
                        placeholder="AIza...",
                        value=saved_api_key,
                        type="password",
                        scale=4,
                    )
                with gr.Row():
                    btn_validate = gr.Button("🔍 키 검증", variant="secondary")
                    btn_save_key = gr.Button("💾 저장", variant="primary")
                key_status = gr.Textbox(label="상태", interactive=False)

                btn_validate.click(on_validate_key, inputs=[api_key_input], outputs=[key_status])
                btn_save_key.click(on_save_key, inputs=[api_key_input], outputs=[key_status])

                gr.Markdown("---")
                gr.Markdown(
                    """
                    ### Kling AI API 키 설정 (영상 생성용)
                    [Kling 개발자 콘솔](https://kling.ai/dev/api-key)에서 Access Key와 Secret Key를 발급받으세요.
                    """
                )
                with gr.Row():
                    kling_access_input = gr.Textbox(
                        label="Kling Access Key",
                        placeholder="Access Key...",
                        value=saved_kling_access,
                        type="password",
                        scale=2,
                    )
                    kling_secret_input = gr.Textbox(
                        label="Kling Secret Key",
                        placeholder="Secret Key...",
                        value=saved_kling_secret,
                        type="password",
                        scale=2,
                    )
                with gr.Row():
                    btn_kling_validate = gr.Button("🔍 키 검증", variant="secondary")
                    btn_kling_save = gr.Button("💾 저장", variant="primary")
                kling_key_status = gr.Textbox(label="Kling 키 상태", interactive=False)

                btn_kling_validate.click(
                    on_validate_kling_keys,
                    inputs=[kling_access_input, kling_secret_input],
                    outputs=[kling_key_status],
                )
                btn_kling_save.click(
                    on_save_kling_keys,
                    inputs=[kling_access_input, kling_secret_input],
                    outputs=[kling_key_status],
                )

            # ── 탭 2: 이미지 생성 ────────────────────────────────────────────
            with gr.Tab("🖼️ 이미지 생성", id="tab_image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 생성 설정")

                        model_radio = gr.Radio(
                            choices=list(MODELS.keys()),
                            value=saved_image_model,
                            label="모델 선택",
                            info="나노 바나나 2: 속도·대량 작업 최적화 / 나노 바나나 프로: 전문 애셋·고급 추론 / 나노 바나나: 초고속·저지연",
                        )

                        ratio_dropdown = gr.Dropdown(
                            choices=list(ASPECT_RATIOS.keys()),
                            value=saved_image_ratio,
                            label="화면 비율",
                        )

                        quality_dropdown = gr.Dropdown(
                            choices=list(QUALITY_OPTIONS.keys()),
                            value=saved_image_quality,
                            label="화질",
                        )

                        count_slider = gr.Slider(
                            minimum=1,
                            maximum=MAX_COUNT,
                            value=saved_image_count,
                            step=1,
                            label=f"생성 개수 (최대 {MAX_COUNT}장)",
                        )

                        ref_image_upload = gr.File(
                            label=f"레퍼런스 이미지 (최대 {MAX_REFERENCE_IMAGES}장)",
                            file_count="multiple",
                            file_types=["image"],
                            elem_id="ref-upload-main",
                            value=saved_ref_paths if saved_ref_paths else None,
                        )

                        ref_image_add = gr.File(
                            label="➕ 레퍼런스 이미지 추가 (현재 목록에 추가됩니다)",
                            file_count="multiple",
                            file_types=["image"],
                            elem_id="ref-upload-add",
                        )

                        btn_clear_refs = gr.Button(
                            "🗑️ 레퍼런스 전체 비우기",
                            variant="secondary",
                            size="sm",
                        )

                        ref_preview = gr.Gallery(
                            label="레퍼런스 미리보기 (썸네일 클릭 시 해당 이미지 삭제 🗑️)",
                            columns=_initial_ref_cols,
                            height=300,
                            object_fit="contain",
                            visible=bool(_initial_ref_preview_imgs),
                            value=_initial_ref_preview_imgs if _initial_ref_preview_imgs else [],
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 프롬프트")

                        prompt_input = gr.Textbox(
                            label="프롬프트",
                            placeholder="생성할 이미지를 설명하세요. 한국어/영어 모두 사용 가능합니다.\n예: A cinematic key visual of a futuristic city at golden hour, dramatic lighting, 8K\n\n💡 Ctrl+Enter로 생성 시작",
                            lines=6,
                            value=saved_image_prompt,
                            elem_id="image-prompt",
                        )

                        model_info = gr.Markdown(
                            f"**{saved_image_model}**: {MODELS[saved_image_model]['description']}"
                        )

                        btn_generate = gr.Button(
                            "🚀 이미지 생성",
                            variant="primary",
                            elem_classes=["generate-btn"],
                            elem_id="image-generate-btn",
                        )

                        gen_status = gr.Textbox(
                            label="상태",
                            interactive=False,
                            value="대기 중",
                            lines=4,
                        )

                # 결과 갤러리 (생성 탭 하단)
                gr.Markdown("### 🖼️ 생성 결과")
                gr.Markdown("💡 이미지 위에 마우스를 올리면 다운로드·영상 레퍼런스·이미지 레퍼런스 버튼이 나타납니다. 여러 장을 선택하려면 왼쪽 위 체크박스를 클릭하세요.")
                with gr.Row():
                    btn_refresh_gen = gr.Button("🔄 새로고침", variant="secondary", scale=1)
                    btn_select_all_gen = gr.Button("☑️ 전체 선택/해제", variant="secondary", scale=1)
                    btn_download_single_gen = gr.Button(
                        "📥 다운로드 (단일 PNG / 다중 선택 시 ZIP)",
                        variant="secondary",
                        scale=2,
                        elem_id="btn-download-gen",
                    )
                    btn_make_video_gen = gr.Button(
                        "🎬 영상 레퍼런스",
                        variant="secondary",
                        scale=1,
                        elem_id="btn-video-gen",
                    )
                    btn_use_as_ref_gen = gr.Button(
                        "🖼️ 이미지 레퍼런스",
                        variant="secondary",
                        scale=1,
                        elem_id="btn-ref-gen",
                    )
                    btn_delete_gen = gr.Button(
                        "🗑️ 선택 삭제",
                        variant="stop",
                        scale=1,
                    )
                live_gallery = gr.Gallery(
                    label="생성된 이미지",
                    columns=4,
                    height=1050,
                    object_fit="contain",
                    value=gallery_state.to_gradio_gallery(),
                    elem_id="live-gallery",
                )
                # 생성 중일 때만 갤러리 상태를 자동 갱신 (idle 시 SSE 불필요 요청 방지)
                gen_timer = gr.Timer(value=1, active=False)
                single_png_output_gen = gr.File(label="PNG 다운로드", elem_id="single-png-gen")
                selected_zip_output_gen = gr.File(label="선택 항목 ZIP 다운로드", elem_id="selected-zip-gen")
                ms_state_gen = gr.Textbox(
                    value="[]",
                    elem_id="ms-state-gen",
                    interactive=True,
                )

                # 오버레이 액션 트리거 (overlay 버튼 → 서버 핸들러 직접 연결용)
                overlay_dl_gen = gr.Textbox(elem_id="overlay-dl-gen", interactive=True)
                overlay_vid_gen = gr.Textbox(elem_id="overlay-vid-gen", interactive=True)
                overlay_ref_gen = gr.Textbox(elem_id="overlay-ref-gen", interactive=True)

                # JS 라이트박스 레퍼런스 패널용: 선택된 이미지의 레퍼런스 절대 경로 JSON (항상 DOM에 유지)
                lb_ref_urls_state = gr.Textbox(value="[]", elem_id="lb-ref-urls-state", interactive=False)

                # 상세보기 레퍼런스 이미지 패널
                with gr.Group(visible=False) as detail_panel_gen:
                    gr.Markdown("#### 🖼️ 이 이미지 생성에 사용된 레퍼런스")
                    detail_ref_gallery_gen = gr.Gallery(
                        label="레퍼런스 이미지",
                        columns=3,
                        height=400,
                        object_fit="contain",
                        elem_id="detail-ref-gen",
                    )


                # 모델 선택 변경 시 설명 업데이트
                def update_model_info(m):
                    return f"**{m}**: {MODELS[m]['description']}"

                def update_ref_visibility(files):
                    if not files:
                        return gr.update(visible=False, value=[])
                    thumbs = []
                    for f in files:
                        try:
                            p = f if isinstance(f, str) else f.path
                            if not p:
                                continue
                            with Image.open(str(p)) as img:
                                img.draft('RGB', (_REF_THUMBNAIL_SIZE, _REF_THUMBNAIL_SIZE))
                                img.thumbnail((_REF_THUMBNAIL_SIZE, _REF_THUMBNAIL_SIZE), Image.LANCZOS)
                                thumbs.append(img.convert("RGB").copy())
                        except Exception:
                            continue
                    cols = 1 if len(thumbs) == 1 else 2
                    return gr.update(visible=bool(thumbs), value=thumbs, columns=cols)

                def on_add_ref_images(new_files, current_files):
                    """추가 업로드된 파일을 기존 레퍼런스 이미지 목록에 병합합니다."""
                    if not new_files:
                        return gr.update(), gr.update(value=None)

                    current_paths = []
                    if current_files:
                        current_paths = [
                            f if isinstance(f, str) else f.path
                            for f in current_files if f is not None
                        ]

                    new_paths = [
                        f if isinstance(f, str) else f.path
                        for f in new_files if f is not None
                    ]

                    total = len(current_paths) + len(new_paths)
                    combined = (current_paths + new_paths)[:MAX_REFERENCE_IMAGES]
                    if total > MAX_REFERENCE_IMAGES:
                        gr.Warning(
                            f"레퍼런스 이미지는 최대 {MAX_REFERENCE_IMAGES}장까지 추가할 수 있습니다. "
                            f"처음 {MAX_REFERENCE_IMAGES}장만 사용됩니다."
                        )

                    return gr.update(value=combined), gr.update(value=None)

                def on_ref_preview_delete(evt: gr.SelectData, current_files):
                    """레퍼런스 미리보기에서 클릭한 이미지를 제거합니다."""
                    idx = evt.index
                    if not current_files:
                        return gr.update(value=None), gr.update(visible=False, value=[])
                    current_paths = [
                        f if isinstance(f, str) else f.path
                        for f in current_files if f is not None
                    ]
                    if not (0 <= idx < len(current_paths)):
                        return gr.update(), gr.update()
                    new_paths = current_paths[:idx] + current_paths[idx + 1:]
                    if not new_paths:
                        return gr.update(value=None), gr.update(visible=False, value=[])
                    thumbs = []
                    for p in new_paths:
                        try:
                            with Image.open(str(p)) as img:
                                img.draft('RGB', (_REF_THUMBNAIL_SIZE, _REF_THUMBNAIL_SIZE))
                                img.thumbnail((_REF_THUMBNAIL_SIZE, _REF_THUMBNAIL_SIZE), Image.LANCZOS)
                                thumbs.append(img.convert("RGB").copy())
                        except Exception:
                            continue
                    cols = 1 if len(thumbs) == 1 else 2
                    return gr.update(value=new_paths), gr.update(visible=bool(thumbs), value=thumbs, columns=cols)

                model_radio.change(update_model_info, inputs=[model_radio], outputs=[model_info]).then(
                    _save_gen_setting("image_model"), inputs=[model_radio]
                )
                ratio_dropdown.change(_save_gen_setting("image_ratio"), inputs=[ratio_dropdown])
                quality_dropdown.change(_save_gen_setting("image_quality"), inputs=[quality_dropdown])
                count_slider.change(_save_gen_setting("image_count"), inputs=[count_slider])
                prompt_input.change(_save_gen_setting("image_prompt"), inputs=[prompt_input])
                ref_image_upload.change(
                    update_ref_visibility,
                    inputs=[ref_image_upload],
                    outputs=[ref_preview],
                ).then(
                    _save_ref_images_to_settings,
                    inputs=[ref_image_upload],
                )
                ref_image_add.change(
                    on_add_ref_images,
                    inputs=[ref_image_add, ref_image_upload],
                    outputs=[ref_image_upload, ref_image_add],
                ).then(
                    update_ref_visibility,
                    inputs=[ref_image_upload],
                    outputs=[ref_preview],
                )
                btn_clear_refs.click(
                    lambda: (gr.update(value=None), gr.update(value=None), gr.update(visible=False, value=[])),
                    outputs=[ref_image_upload, ref_image_add, ref_preview],
                )
                ref_preview.select(
                    on_ref_preview_delete,
                    inputs=[ref_image_upload],
                    outputs=[ref_image_upload, ref_preview],
                )

                generate_fn = build_generate_fn(gallery_state)

                btn_generate.click(
                    generate_fn,
                    inputs=[
                        model_radio,
                        prompt_input,
                        ratio_dropdown,
                        quality_dropdown,
                        count_slider,
                        ref_image_upload,
                    ],
                    outputs=[live_gallery, gen_status],
                    concurrency_limit=None,
                ).then(
                    lambda: gr.update(active=True),
                    outputs=[gen_timer],
                    concurrency_limit=None,
                ).then(
                    update_ref_visibility,
                    inputs=[ref_image_upload],
                    outputs=[ref_preview],
                )

                # 타이머 갱신: 백그라운드 생성이 진행되는 동안 갤러리와 상태를 자동으로 업데이트
                def poll_gallery_state():
                    # pending 항목이 없으면 타이머 비활성화 (idle 시 SSE 요청 제거)
                    if not gallery_state.has_pending():
                        return gr.update(), gr.update(), gr.update(active=False)
                    return gallery_state.to_gradio_gallery(), gallery_state.get_summary(), gr.update()

                gen_timer.tick(poll_gallery_state, outputs=[live_gallery, gen_status, gen_timer])

                def _load_ref_detail_images(item):
                    """GalleryItem에서 레퍼런스 이미지를 상세 패널용 크기로 로드합니다."""
                    if item is None or not item.reference_image_paths:
                        return [], False
                    imgs = []
                    for rp in item.reference_image_paths:
                        try:
                            if os.path.exists(rp):
                                img = Image.open(rp).convert("RGB")
                                img.thumbnail((512, 512), Image.LANCZOS)
                                imgs.append(img)
                        except Exception:
                            continue
                    return imgs, bool(imgs)

                def on_gen_gallery_select(evt: gr.SelectData):
                    idx = evt.index
                    item = gallery_state.get_success_item_by_visual_index(idx)
                    ref_imgs, has_refs = _load_ref_detail_images(item)
                    # JS 라이트박스 패널용: 절대 경로 JSON (Gradio 파일 API로 접근 가능)
                    ref_abs_paths = []
                    if item and item.reference_image_paths:
                        for p in item.reference_image_paths:
                            try:
                                abs_p = str(Path(p).resolve())
                                if os.path.exists(abs_p):
                                    ref_abs_paths.append(abs_p)
                            except (OSError, ValueError):
                                pass
                    return idx, ref_imgs, gr.update(visible=False), json.dumps(ref_abs_paths)

                live_gallery.select(
                    on_gen_gallery_select,
                    outputs=[selected_img_idx_gen, detail_ref_gallery_gen, detail_panel_gen, lb_ref_urls_state],
                )

                def refresh_live_gallery():
                    return gallery_state.to_gradio_gallery()

                btn_refresh_gen.click(refresh_live_gallery, outputs=[live_gallery])

                smart_download_gen_fn = build_smart_download_fn(gallery_state)
                btn_download_single_gen.click(
                    smart_download_gen_fn,
                    inputs=[ms_state_gen, selected_img_idx_gen],
                    outputs=[single_png_output_gen],
                    js="(ms, idx) => { var wasOv = typeof window.__ovIdx !== 'undefined' && window.__ovIdx >= 0; var oi = wasOv ? window.__ovIdx : idx; window.__ovIdx = -1; return wasOv ? [JSON.stringify([oi]), oi] : [window.__getSelJson('live-gallery', ms), oi]; }",
                )

                btn_use_as_ref_gen.click(
                    _use_as_ref_selected,
                    inputs=[ms_state_gen, ref_image_upload, selected_img_idx_gen],
                    outputs=[ref_image_upload],
                    js="(ms, files, idx) => { var wasOv = typeof window.__ovIdx !== 'undefined' && window.__ovIdx >= 0; var oi = wasOv ? window.__ovIdx : idx; window.__ovIdx = -1; return wasOv ? [JSON.stringify([oi]), files, oi] : [window.__getSelJson('live-gallery', ms), files, oi]; }",
                )

                delete_gen_fn = build_delete_selected_fn(gallery_state)
                btn_delete_gen.click(
                    delete_gen_fn,
                    inputs=[ms_state_gen, selected_img_idx_gen],
                    outputs=[live_gallery, gen_status, ms_state_gen, selected_img_idx_gen],
                    js="(ms, idx) => [window.__getSelJson('live-gallery', ms), idx]",
                )

                btn_select_all_gen.click(
                    fn=None,
                    js="() => { if (window.__msToggleAll) window.__msToggleAll('live-gallery'); }",
                )

                # 오버레이 액션 핸들러 연결
                def _parse_overlay_idx(val: str) -> int:
                    """'idx:timestamp' 형식 또는 단순 숫자에서 인덱스 추출"""
                    try:
                        return int(val.split(":")[0])
                    except Exception:
                        return -1

                # overlay_dl_gen 은 Python 핸들러로 원본 사이즈 PNG를 반환 (탭 블록 외부에서 등록)
                # overlay_vid_gen / overlay_ref_gen 은 triggerOverlayAction → Python 핸들러 필요
                # (핸들러는 ref_image_vid 정의 후 탭 블록 외부에서 등록)

            # ── 탭 3: 영상 생성 (Kling) ──────────────────────────────────────
            with gr.Tab("🎬 영상 생성 (Kling)", id="tab_video"):
                gr.Markdown(
                    """
                    ### 🎬 Kling AI 영상 생성
                    입력 방식을 선택하고 설정을 조정한 뒤 **영상 생성** 버튼을 클릭하세요.
                    Kling API 키는 **API 키 설정** 탭에서 저장할 수 있습니다.
                    """
                )

                # 입력 모드 상태: "image" | "start_end" | "video_ref"
                video_mode_state = gr.State("image")

                with gr.Row():
                    with gr.Column(scale=1):

                        gr.Markdown("#### 📥 입력 방식 선택")
                        with gr.Tabs() as video_input_tabs:

                            with gr.Tab("🖼️ 레퍼런스 이미지"):
                                gr.Markdown("시작 프레임 역할을 할 이미지를 업로드하세요. **이미지가 없으면 프롬프트만으로 텍스트→영상을 생성합니다.**")
                                ref_image_vid = gr.Image(
                                    label="레퍼런스 이미지 (갤러리에서 전송 또는 직접 업로드)",
                                    type="pil",
                                    height=200,
                                )

                            with gr.Tab("🎞️ 시작-끝 프레임"):
                                gr.Markdown("시작 프레임과 끝 프레임을 각각 업로드하면 두 이미지 사이를 자연스럽게 이어주는 영상이 생성됩니다.")
                                start_image_vid = gr.Image(
                                    label="시작 프레임",
                                    type="pil",
                                    height=180,
                                )
                                end_image_vid = gr.Image(
                                    label="끝 프레임",
                                    type="pil",
                                    height=180,
                                )

                            with gr.Tab("🎬 영상 레퍼런스", visible=False) as video_ref_tab:
                                gr.Markdown("3~10초 분량의 레퍼런스 영상을 업로드하세요. 영상의 스타일·움직임을 참고하여 새로운 영상을 생성합니다.\n\n⚠️ **Kling 3 Omni 모델에서만 사용 가능합니다.**")
                                ref_video_vid = gr.Video(
                                    label="레퍼런스 영상 (3~10초)",
                                    height=200,
                                )

                        def _on_input_mode_change(evt: gr.SelectData):
                            modes = ["image", "start_end", "video_ref"]
                            return modes[evt.index] if 0 <= evt.index < len(modes) else "image"

                        video_input_tabs.select(
                            _on_input_mode_change,
                            outputs=[video_mode_state],
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("#### ⚙️ 영상 설정")

                        kling_model_radio = gr.Radio(
                            choices=list(KLING_MODELS.keys()),
                            value=saved_video_model,
                            label="Kling 모델",
                            info=" | ".join(
                                f"{k}: {v['description']}"
                                for k, v in KLING_MODELS.items()
                            ),
                        )

                        kling_quality_dropdown = gr.Dropdown(
                            choices=list(KLING_QUALITY_OPTIONS.keys()),
                            value=saved_video_quality,
                            label="화질",
                            info="720p (Standard): 빠름 / 1080p (Professional): 고화질",
                        )

                        kling_duration_slider = gr.Slider(
                            minimum=3,
                            maximum=15,
                            value=saved_video_duration,
                            step=1,
                            label="영상 길이 (초)",
                            info="3초부터 15초까지 1초 단위로 선택",
                        )

                        kling_ratio_dropdown = gr.Dropdown(
                            choices=KLING_VIDEO_RATIOS,
                            value=saved_video_ratio,
                            label="화면 비율",
                        )

                        kling_prompt_input = gr.Textbox(
                            label="프롬프트 (선택)",
                            placeholder="영상 움직임을 설명하세요. 예: Camera slowly zooms in, dramatic lighting\n\n💡 Ctrl+Enter로 생성 시작",
                            lines=3,
                            value=saved_video_prompt,
                            elem_id="video-prompt",
                        )

                        enable_audio_checkbox = gr.Checkbox(
                            label="🔊 오디오 생성 사용 (Kling v3 / v3 Omni 지원)",
                            value=saved_video_enable_audio,
                            info="Kling v3 및 Kling 3 Omni 모델에서 오디오를 함께 생성합니다.",
                        )

                        btn_generate_video = gr.Button(
                            "🎬 영상 생성",
                            variant="primary",
                            elem_classes=["generate-btn"],
                            elem_id="video-generate-btn",
                        )

                        video_status = gr.Textbox(
                            label="상태",
                            interactive=False,
                            value="대기 중",
                        )

                gr.Markdown("### 🎥 생성 결과")
                gr.Markdown("💡 영상 썸네일 위에 마우스를 올리면 다운로드 버튼이 나타납니다. 여러 개를 선택하려면 왼쪽 위 체크박스를 클릭하세요. 썸네일을 클릭하면 아래 영상 플레이어에서 재생됩니다.")

                with gr.Row():
                    btn_refresh_videos = gr.Button("🔄 새로고침", variant="secondary", scale=1)
                    btn_select_all_video = gr.Button("☑️ 전체 선택/해제", variant="secondary", scale=1)
                    btn_download_single_video = gr.Button(
                        "📥 다운로드 (단일 MP4 / 다중 선택 시 ZIP)",
                        variant="secondary",
                        scale=2,
                        elem_id="btn-download-video",
                    )
                    btn_delete_video = gr.Button(
                        "🗑️ 선택 삭제",
                        variant="stop",
                        scale=1,
                    )

                video_gallery = gr.Gallery(
                    label="생성된 영상",
                    columns=4,
                    height=600,
                    object_fit="contain",
                    value=video_gallery_state.to_gradio_gallery(),
                    elem_id="video-gallery",
                )

                video_player = gr.Video(
                    label="선택한 영상 재생",
                    height=480,
                    visible=False,
                )

                video_timer = gr.Timer(value=2, active=False)
                single_video_output = gr.File(label="MP4 다운로드", elem_id="single-video-gen")
                selected_videos_zip_output_gen = gr.File(label="선택 영상 ZIP 다운로드", elem_id="selected-videos-zip-gen")
                ms_state_video = gr.Textbox(
                    value="[]",
                    elem_id="ms-state-video",
                    interactive=True,
                )
                selected_vid_idx = gr.State(-1)
                overlay_dl_video = gr.Textbox(elem_id="overlay-dl-video", interactive=True)

                # ── 이벤트 핸들러 등록 ─────────────────────────────────────────

                def _on_kling_model_change(model_label: str):
                    """Omni 모델(supports_video_reference=True)에서만 영상 레퍼런스 탭 표시"""
                    supports = KLING_MODELS.get(model_label, {}).get("supports_video_reference", False)
                    return gr.update(visible=supports)

                kling_model_radio.change(
                    _on_kling_model_change,
                    inputs=[kling_model_radio],
                    outputs=[video_ref_tab],
                ).then(
                    _save_gen_setting("video_model"), inputs=[kling_model_radio]
                )
                kling_quality_dropdown.change(_save_gen_setting("video_quality"), inputs=[kling_quality_dropdown])
                kling_duration_slider.change(_save_gen_setting("video_duration"), inputs=[kling_duration_slider])
                kling_ratio_dropdown.change(_save_gen_setting("video_ratio"), inputs=[kling_ratio_dropdown])
                kling_prompt_input.change(_save_gen_setting("video_prompt"), inputs=[kling_prompt_input])
                enable_audio_checkbox.change(_save_gen_setting("video_enable_audio"), inputs=[enable_audio_checkbox])

                unified_video_fn = build_unified_video_fn(gallery_state, video_gallery_state)
                btn_generate_video.click(
                    unified_video_fn,
                    inputs=[
                        video_mode_state,
                        ref_image_vid,
                        start_image_vid,
                        end_image_vid,
                        ref_video_vid,
                        kling_prompt_input,
                        kling_model_radio,
                        kling_duration_slider,
                        kling_ratio_dropdown,
                        enable_audio_checkbox,
                        kling_quality_dropdown,
                    ],
                    outputs=[video_player, video_status],
                ).then(
                    lambda: (gr.update(active=True), gr.update(visible=True)),
                    outputs=[video_timer, video_player],
                )

                def on_video_gallery_select(evt: gr.SelectData):
                    idx = evt.index
                    item = video_gallery_state.get_item_by_visual_index(idx)
                    if item is None:
                        return idx, gr.update(value=None, visible=False)
                    if _is_safe_output_path(item.video_path) and os.path.exists(item.video_path):
                        return idx, gr.update(value=item.video_path, visible=True)
                    return idx, gr.update(value=None, visible=False)

                video_gallery.select(
                    on_video_gallery_select,
                    outputs=[selected_vid_idx, video_player],
                )

                def poll_video_gallery_state():
                    if not video_gallery_state.has_pending():
                        return gr.update(), gr.update(), gr.update(active=False)
                    return (
                        video_gallery_state.to_gradio_gallery(),
                        video_gallery_state.get_summary(),
                        gr.update(),
                    )

                video_timer.tick(poll_video_gallery_state, outputs=[video_gallery, video_status, video_timer])

                def refresh_video_gallery_fn():
                    return (
                        video_gallery_state.to_gradio_gallery(),
                        video_gallery_state.get_summary(),
                    )

                btn_refresh_videos.click(
                    refresh_video_gallery_fn,
                    outputs=[video_gallery, video_status],
                )

                def smart_download_video(selected_json: str, current_idx: int):
                    """단일 선택 → MP4, 다중 선택 → ZIP 스마트 다운로드"""
                    try:
                        indices = json.loads(selected_json or "[]")
                    except (json.JSONDecodeError, ValueError):
                        indices = []

                    if len(indices) > 1:
                        paths = []
                        for raw_idx in indices:
                            try:
                                item = video_gallery_state.get_item_by_visual_index(int(raw_idx))
                                if item and item.video_path and os.path.exists(item.video_path):
                                    paths.append(item.video_path)
                            except (ValueError, TypeError):
                                continue
                        if not paths:
                            gr.Warning("선택된 영상 파일을 찾을 수 없습니다.")
                            return None
                        return create_zip_from_paths(paths)
                    elif len(indices) == 1:
                        try:
                            item = video_gallery_state.get_item_by_visual_index(int(indices[0]))
                        except (ValueError, TypeError):
                            item = None
                    else:
                        item = video_gallery_state.get_item_by_visual_index(current_idx)

                    if item is None:
                        gr.Warning("영상을 먼저 클릭하여 선택해주세요.")
                        return None
                    if not item.video_path or not os.path.exists(item.video_path):
                        gr.Warning("영상 파일을 찾을 수 없습니다.")
                        return None
                    return item.video_path

                btn_download_single_video.click(
                    smart_download_video,
                    inputs=[ms_state_video, selected_vid_idx],
                    outputs=[single_video_output],
                    js="(ms, idx) => { var wasOv = typeof window.__ovIdx !== 'undefined' && window.__ovIdx >= 0; var oi = wasOv ? window.__ovIdx : idx; window.__ovIdx = -1; return wasOv ? [JSON.stringify([oi]), oi] : [window.__getSelJson('video-gallery', ms), oi]; }",
                )

                def delete_selected_videos(selected_json: str, current_idx: int):
                    try:
                        indices = json.loads(selected_json or "[]")
                    except (json.JSONDecodeError, ValueError):
                        indices = []
                    if not indices and current_idx >= 0:
                        indices = [current_idx]
                    if not indices:
                        gr.Warning("삭제할 영상을 먼저 선택해주세요.")
                        return (
                            video_gallery_state.to_gradio_gallery(),
                            video_gallery_state.get_summary(),
                            "[]",
                            -1,
                            gr.update(value=None, visible=False),
                        )

                    removed_paths = video_gallery_state.remove_by_visual_indices(
                        [int(i) for i in indices]
                    )
                    for path in removed_paths:
                        if _is_safe_output_path(path):
                            try:
                                if os.path.exists(path):
                                    os.remove(path)
                                meta_path = path.replace(".mp4", ".json")
                                if os.path.exists(meta_path):
                                    os.remove(meta_path)
                                from core.image_utils import get_video_thumbnail_path as _gvtp
                                thumb = _gvtp(path)
                                if os.path.exists(thumb):
                                    os.remove(thumb)
                            except Exception:
                                pass

                    deleted_count = len(removed_paths)
                    return (
                        video_gallery_state.to_gradio_gallery(),
                        video_gallery_state.get_summary() + f" (🗑️ {deleted_count}개 삭제됨)",
                        "[]",
                        -1,
                        gr.update(value=None, visible=False),
                    )

                btn_delete_video.click(
                    delete_selected_videos,
                    inputs=[ms_state_video, selected_vid_idx],
                    outputs=[video_gallery, video_status, ms_state_video, selected_vid_idx, video_player],
                    js="(ms, idx) => [window.__getSelJson('video-gallery', ms), idx]",
                )

                btn_select_all_video.click(
                    fn=None,
                    js="() => { if (window.__msToggleAll) window.__msToggleAll('video-gallery'); }",
                )

            # ── 탭 4: 갤러리 & 다운로드 ──────────────────────────────────────
            with gr.Tab("📁 갤러리 & 다운로드", id="tab_gallery"):
                gr.Markdown("### 생성된 이미지 전체 보기 및 다운로드")

                disk_usage_md = gr.Markdown(get_disk_usage_text())

                with gr.Row():
                    btn_refresh = gr.Button("🔄 갤러리 새로고침", variant="secondary")
                    btn_download_zip = gr.Button("⬇️ 전체 ZIP 다운로드", variant="primary")
                    btn_clear = gr.Button("🗑️ 갤러리 초기화", variant="stop")

                gallery_status = gr.Textbox(
                    label="상태",
                    value=gallery_state.get_summary(),
                    interactive=False,
                )

                with gr.Row():
                    btn_select_all_gallery = gr.Button("☑️ 전체 선택/해제", variant="secondary", scale=1)
                    btn_download_single_gallery = gr.Button(
                        "📥 다운로드 (단일 PNG / 다중 선택 시 ZIP)",
                        variant="secondary",
                        scale=2,
                        elem_id="btn-download-gallery",
                    )
                    btn_make_video_gallery = gr.Button(
                        "🎬 영상 레퍼런스",
                        variant="secondary",
                        scale=1,
                        elem_id="btn-video-gallery",
                    )
                    btn_use_as_ref_gallery = gr.Button(
                        "🖼️ 이미지 레퍼런스",
                        variant="secondary",
                        scale=1,
                        elem_id="btn-ref-gallery",
                    )
                    btn_delete_gallery = gr.Button(
                        "🗑️ 선택 삭제",
                        variant="stop",
                        scale=1,
                    )

                full_gallery = gr.Gallery(
                    label="전체 갤러리",
                    columns=4,
                    height=900,
                    object_fit="contain",
                    value=gallery_state.to_gradio_gallery(),
                    elem_id="full-gallery",
                )

                single_png_output_gallery = gr.File(label="PNG 다운로드", elem_id="single-png-gallery")
                selected_zip_output_gallery = gr.File(label="선택 항목 ZIP 다운로드", elem_id="selected-zip-gallery")
                zip_file_output = gr.File(label="ZIP 다운로드", elem_id="full-zip-gallery")
                ms_state_gallery = gr.Textbox(
                    value="[]",
                    elem_id="ms-state-gallery",
                    interactive=True,
                )

                # 오버레이 액션 트리거 (갤러리 탭)
                overlay_dl_gallery = gr.Textbox(elem_id="overlay-dl-gallery", interactive=True)
                overlay_vid_gallery = gr.Textbox(elem_id="overlay-vid-gallery", interactive=True)
                overlay_ref_gallery = gr.Textbox(elem_id="overlay-ref-gallery", interactive=True)

                # 상세보기 레퍼런스 이미지 패널 (갤러리 탭)
                with gr.Group(visible=False) as detail_panel_gallery:
                    gr.Markdown("#### 🖼️ 이 이미지 생성에 사용된 레퍼런스")
                    detail_ref_gallery_full = gr.Gallery(
                        label="레퍼런스 이미지",
                        columns=3,
                        height=400,
                        object_fit="contain",
                        elem_id="detail-ref-full",
                    )


                def refresh_gallery():
                    return (
                        gallery_state.to_gradio_gallery(),
                        gallery_state.get_summary(),
                        get_disk_usage_text(),
                    )

                btn_refresh.click(refresh_gallery, outputs=[full_gallery, gallery_status, disk_usage_md])

                download_zip_fn = build_download_zip_fn(gallery_state)
                btn_download_zip.click(download_zip_fn, outputs=[zip_file_output])

                clear_fn = build_clear_fn(gallery_state)
                btn_clear.click(clear_fn, outputs=[full_gallery, gallery_status, zip_file_output])

                def on_gallery_tab_select(evt: gr.SelectData):
                    idx = evt.index
                    item = gallery_state.get_success_item_by_visual_index(idx)
                    ref_imgs, has_refs = _load_ref_detail_images(item)
                    # JS 라이트박스 패널용: 절대 경로 JSON (Gradio 파일 API로 접근 가능)
                    ref_abs_paths = []
                    if item and item.reference_image_paths:
                        for p in item.reference_image_paths:
                            try:
                                abs_p = str(Path(p).resolve())
                                if os.path.exists(abs_p):
                                    ref_abs_paths.append(abs_p)
                            except (OSError, ValueError):
                                pass
                    return idx, ref_imgs, gr.update(visible=False), json.dumps(ref_abs_paths)

                full_gallery.select(
                    on_gallery_tab_select,
                    outputs=[selected_img_idx_gallery, detail_ref_gallery_full, detail_panel_gallery, lb_ref_urls_state],
                )

                smart_download_gallery_fn = build_smart_download_fn(gallery_state)
                btn_download_single_gallery.click(
                    smart_download_gallery_fn,
                    inputs=[ms_state_gallery, selected_img_idx_gallery],
                    outputs=[single_png_output_gallery],
                    js="(ms, idx) => { var wasOv = typeof window.__ovIdx !== 'undefined' && window.__ovIdx >= 0; var oi = wasOv ? window.__ovIdx : idx; window.__ovIdx = -1; return wasOv ? [JSON.stringify([oi]), oi] : [window.__getSelJson('full-gallery', ms), oi]; }",
                )

                btn_use_as_ref_gallery.click(
                    _use_as_ref_selected,
                    inputs=[ms_state_gallery, ref_image_upload, selected_img_idx_gallery],
                    outputs=[ref_image_upload],
                    js="(ms, files, idx) => { var wasOv = typeof window.__ovIdx !== 'undefined' && window.__ovIdx >= 0; var oi = wasOv ? window.__ovIdx : idx; window.__ovIdx = -1; return wasOv ? [JSON.stringify([oi]), files, oi] : [window.__getSelJson('full-gallery', ms), files, oi]; }",
                )

                delete_gallery_fn = build_delete_selected_fn(gallery_state)
                btn_delete_gallery.click(
                    delete_gallery_fn,
                    inputs=[ms_state_gallery, selected_img_idx_gallery],
                    outputs=[full_gallery, gallery_status, ms_state_gallery, selected_img_idx_gallery],
                    js="(ms, idx) => [window.__getSelJson('full-gallery', ms), idx]",
                )

                btn_select_all_gallery.click(
                    fn=None,
                    js="() => { if (window.__msToggleAll) window.__msToggleAll('full-gallery'); }",
                )

                # overlay_dl_gallery 는 Python 핸들러로 원본 사이즈 PNG를 반환 (탭 블록 외부에서 등록)
                # overlay_vid_gallery / overlay_ref_gallery 는 탭 블록 외부에서 핸들러 등록

        # ── 탭 간 "영상화" 버튼 공통 핸들러 ─────────────────────────────────

        def _load_image_for_video(item: Optional[GalleryItem]) -> Optional[Image.Image]:
            """GalleryItem에서 원본 이미지를 로드합니다."""
            if item is None:
                return None
            try:
                if item.image_path and os.path.exists(item.image_path):
                    return Image.open(item.image_path).convert("RGB")
            except Exception:
                pass
            return item.image

        def on_make_video_selected(selected_json: str, fallback_idx: int):
            """체크박스로 선택된 첫 번째 이미지를 영상 생성 탭으로 전송합니다."""
            try:
                indices = json.loads(selected_json or "[]")
            except (json.JSONDecodeError, ValueError):
                indices = []
            idx = int(indices[0]) if indices else fallback_idx
            item = gallery_state.get_success_item_by_visual_index(idx)
            if item is None:
                gr.Warning("이미지를 먼저 선택해주세요.")
                return None, gr.update()
            img = _load_image_for_video(item)
            return img, gr.update(selected="tab_video")

        def on_make_video_single(val: str):
            """오버레이 영상화 버튼: 단일 이미지를 영상 생성 탭으로 전송합니다."""
            try:
                idx = int(val.split(":")[0])
            except Exception:
                idx = -1
            item = gallery_state.get_success_item_by_visual_index(idx)
            if item is None:
                gr.Warning("이미지를 먼저 클릭하여 선택해주세요.")
                return None, gr.update()
            img = _load_image_for_video(item)
            return img, gr.update(selected="tab_video")

        btn_make_video_gen.click(
            on_make_video_selected,
            inputs=[ms_state_gen, selected_img_idx_gen],
            outputs=[ref_image_vid, main_tabs],
            js="(ms, idx) => { var wasOv = typeof window.__ovIdx !== 'undefined' && window.__ovIdx >= 0; var oi = wasOv ? window.__ovIdx : idx; window.__ovIdx = -1; return wasOv ? [JSON.stringify([oi]), oi] : [window.__getSelJson('live-gallery', ms), oi]; }",
        )

        btn_make_video_gallery.click(
            on_make_video_selected,
            inputs=[ms_state_gallery, selected_img_idx_gallery],
            outputs=[ref_image_vid, main_tabs],
            js="(ms, idx) => { var wasOv = typeof window.__ovIdx !== 'undefined' && window.__ovIdx >= 0; var oi = wasOv ? window.__ovIdx : idx; window.__ovIdx = -1; return wasOv ? [JSON.stringify([oi]), oi] : [window.__getSelJson('full-gallery', ms), oi]; }",
        )

        # 오버레이 영상화 버튼 핸들러 (개별 이미지 기준, triggerOverlayAction 경유)
        overlay_vid_gen.input(
            on_make_video_single,
            inputs=[overlay_vid_gen],
            outputs=[ref_image_vid, main_tabs],
        )

        overlay_vid_gallery.input(
            on_make_video_single,
            inputs=[overlay_vid_gallery],
            outputs=[ref_image_vid, main_tabs],
        )

        # 오버레이 레퍼런스 버튼 핸들러 (개별 이미지 기준, triggerOverlayAction 경유)
        overlay_ref_gen.input(
            _use_as_ref_single,
            inputs=[overlay_ref_gen, ref_image_upload],
            outputs=[ref_image_upload],
        )

        overlay_ref_gallery.input(
            _use_as_ref_single,
            inputs=[overlay_ref_gallery, ref_image_upload],
            outputs=[ref_image_upload],
        )

        # 오버레이 다운로드 버튼 핸들러: 원본 사이즈 PNG를 서버에서 반환 (triggerOverlayAction 경유)
        def _overlay_download(val: str):
            """'idx:timestamp' 형식에서 인덱스를 추출하여 원본 이미지 경로를 반환합니다."""
            try:
                idx = int(val.split(":")[0])
            except Exception:
                return None
            item = gallery_state.get_success_item_by_visual_index(idx)
            if item is None or not item.image_path or not os.path.exists(item.image_path):
                return None
            return item.image_path

        overlay_dl_gen.input(
            _overlay_download,
            inputs=[overlay_dl_gen],
            outputs=[single_png_output_gen],
        )

        overlay_dl_gallery.input(
            _overlay_download,
            inputs=[overlay_dl_gallery],
            outputs=[single_png_output_gallery],
        )

        # 오버레이 다운로드 버튼 핸들러: 영상 갤러리 (triggerOverlayAction 경유)
        def _overlay_download_video(val: str):
            """'idx:timestamp' 형식에서 인덱스를 추출하여 영상 경로를 반환합니다."""
            try:
                idx = int(val.split(":")[0])
            except Exception:
                return None
            item = video_gallery_state.get_item_by_visual_index(idx)
            if item is None or not item.video_path or not os.path.exists(item.video_path):
                return None
            return item.video_path

        overlay_dl_video.input(
            _overlay_download_video,
            inputs=[overlay_dl_video],
            outputs=[single_video_output],
        )

        gr.Markdown(
            """
            ---
            **안내**: 생성된 이미지와 영상은 `outputs/YYYY-MM-DD/` 폴더에 자동 저장됩니다.
            나노 바나나 2와 나노 바나나 프로 모두 레퍼런스 이미지를 지원합니다.
            Kling 영상 생성은 보통 1~5분 정도 소요됩니다.
            """
        )

    return demo
