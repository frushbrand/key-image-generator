"""
Gradio UI 컴포넌트 및 이벤트 핸들러 모듈
"""

import json
import os
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
    PLACEHOLDER_IMAGE_PATH,
    ensure_placeholder_image,
    load_existing_outputs,
    load_existing_video_outputs,
    get_disk_usage_text,
)
from ui.gallery import GalleryState, GalleryItem


# ── 앱 테마 및 CSS (Gradio 6.0+: launch()에 전달) ───────────────────────────

APP_THEME = gr.themes.Soft()

APP_CSS = """
        .title-text { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }
        .subtitle-text { color: #666; margin-bottom: 1rem; }
        .generate-btn { background: #4f46e5 !important; color: white !important; font-size: 1.1rem !important; }

        /* 갤러리 라이트박스/이미지 뷰어 버튼 크기 확대 */
        .lightbox button,
        .lightbox .icon-button,
        [data-testid="lightbox"] button,
        .gallery-container .icon-button {
            width: 48px !important;
            height: 48px !important;
            font-size: 1.4rem !important;
            min-width: 48px !important;
        }
        .lightbox svg,
        [data-testid="lightbox"] svg,
        .gallery-container button svg {
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

        /* 갤러리 캡션(라이트박스 포함) 전체 표시 - 잘림 방지 */
        [data-testid="caption"],
        .gallery-item .caption,
        [class*="caption"] {
            white-space: pre-wrap !important;
            overflow: visible !important;
            text-overflow: unset !important;
            display: block !important;
            max-height: none !important;
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

        /* 파일 다운로드 출력 위젯: MutationObserver 감지를 위해 DOM에 존재하되 숨김
           Gradio 5는 visible=False 시 DOM에서 완전히 제거(Svelte 조건부 렌더링)하므로
           JS에서 접근해야 하는 컴포넌트는 visible=True + CSS hidden 방식 사용 */
        #single-png-gen, #selected-zip-gen,
        #single-png-gallery, #full-zip-gallery, #selected-zip-gallery,
        #selected-videos-zip, #all-videos-zip {
            display: none !important;
        }
        /* 내부 상태/트리거 텍스트박스: display:none 대신 화면 밖 배치 사용
           display:none 시 Gradio 5 Svelte 이벤트 파이프라인이 synthetic event를
           처리하지 않아 Python 핸들러가 트리거되지 않음 */
        #ms-state-gen, #ms-state-gallery,
        #overlay-dl-gen, #overlay-vid-gen, #overlay-ref-gen,
        #overlay-dl-gallery, #overlay-vid-gallery, #overlay-ref-gallery {
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


def build_unified_video_fn(gallery_state: GalleryState):
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

        progress(0.05, desc="영상 생성 작업 요청 중...")
        endpoint = "image2video"

        try:
            if mode == "start_end":
                if start_image is None or end_image is None:
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
            return None, f"⏱️ {e}"
        except Exception as e:
            return None, f"❌ 생성 실패: {e}"

        progress(0.97, desc="영상 다운로드 중...")
        try:
            video_bytes = download_video(video_url)
        except Exception as e:
            return None, f"❌ 영상 다운로드 실패: {e}"

        video_path = save_video(video_bytes, model_label, prompt or "")
        progress(1.0, desc="완료!")
        return video_path, f"✅ 영상 생성 완료! 저장 위치: {video_path}"

    return generate_video


def build_generate_fn(gallery_state: GalleryState):
    """생성 버튼 클릭 핸들러 팩토리 (제너레이터 방식: 이미지 완성 즉시 갤러리 업데이트)"""

    def generate(
        model_name: str,
        prompt: str,
        ratio: str,
        quality: str,
        count: int,
        ref_images,          # Gradio File 컴포넌트 값 (list of paths or None)
    ):
        import queue as _queue
        import threading as _threading
        import time as _time

        cfg = load_settings()
        api_key = cfg.get("api_key", "")
        if not api_key or not api_key.strip():
            gr.Warning("API 키 설정 탭에서 Google API 키를 먼저 저장해주세요.")
            yield gallery_state.to_gradio_gallery(), gallery_state.get_summary()
            return

        if not prompt or not prompt.strip():
            gr.Warning("프롬프트를 입력해주세요.")
            yield gallery_state.to_gradio_gallery(), gallery_state.get_summary()
            return

        count = max(1, min(int(count), MAX_COUNT))

        # 레퍼런스 이미지 로드
        ref_pil_images: list[Image.Image] = []
        ref_paths: list[str] = []
        if ref_images:
            ref_paths = [r if isinstance(r, str) else r.path for r in ref_images if r is not None]
            ref_pil_images = load_reference_images(ref_paths[:MAX_REFERENCE_IMAGES])
            ref_paths = ref_paths[:MAX_REFERENCE_IMAGES]

        # ① 즉시 N개의 대기 중 슬롯 생성
        allocated_indices = gallery_state.allocate_pending_items(
            count, model_name, ratio, quality, prompt
        )

        # ② 갤러리를 즉시 업데이트 (플레이스홀더 표시)
        yield gallery_state.to_gradio_gallery(), f"⏳ {count}장 생성 시작..."

        # ③ 완료 결과를 주고받는 큐
        result_queue: _queue.Queue = _queue.Queue()

        def on_progress(idx: int, img, err):
            result_queue.put((idx, img, err))

        start_time = _time.time()

        # ④ 백그라운드 스레드에서 실제 이미지 생성
        def run_generation():
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
            finally:
                result_queue.put(None)  # 완료 센티널

        gen_thread = _threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        # ⑤ 결과가 도착할 때마다 해당 슬롯을 채우고 갤러리를 즉시 업데이트
        finished = 0
        failed_errors: list[str] = []

        while finished < count:
            try:
                result = result_queue.get(timeout=600)  # 10분 타임아웃
            except _queue.Empty:
                break

            if result is None:
                break

            idx, img, err = result
            gallery_index = allocated_indices[idx]

            if img is not None:
                img_path, _, thumb_path = save_image(
                    img, model_name, ratio, prompt, quality,
                    reference_image_paths=ref_paths,
                )
                gallery_state.fill_pending_item(gallery_index, img_path, thumb_path, "success")
            else:
                gallery_state.fill_pending_item(gallery_index, "", "", "failed", err)
                failed_errors.append(f"  • #{gallery_index + 1}: {err}")

            finished += 1
            elapsed = int(_time.time() - start_time)
            yield (
                gallery_state.to_gradio_gallery(),
                f"⏳ 생성 중... {finished}/{count}장 완료 | ⏱️ {elapsed}초",
            )

        gen_thread.join(timeout=1)

        elapsed_total = int(_time.time() - start_time)
        status_msg = gallery_state.get_summary() + f" (⏱️ {elapsed_total}초 소요)"
        if failed_errors:
            status_msg += "\n\n❌ 실패 원인:\n" + "\n".join(failed_errors)

        yield gallery_state.to_gradio_gallery(), status_msg

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
        return src.replace(/\/thumbs\/([^/?]+)(\?|$)/, '/$1$2');
    }

    // ── 갤러리 호버 플로팅 오버레이 ─────────────────────────────────────────
    (function() {
        var CFGS = [
            {id:'live-gallery', dl:'btn-download-gen',     vid:'btn-video-gen',     ref:'btn-ref-gen'},
            {id:'full-gallery', dl:'btn-download-gallery', vid:'btn-video-gallery', ref:'btn-ref-gallery'}
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

        function cfgOf(el) {
            for (var i = 0; i < CFGS.length; i++) {
                var g = document.getElementById(CFGS[i].id);
                if (g && g.contains(el)) return CFGS[i];
            }
            return null;
        }
        function itemOf(el) {
            return el.closest('[data-testid="thumbnail"]')
                || el.closest('.thumbnail-item')
                || el.closest('[class*="thumbnail-item"]');
        }
        function getItems(cfg) {
            var g = document.getElementById(cfg.id); if (!g) return [];
            var r = Array.from(g.querySelectorAll('[data-testid="thumbnail"]'));
            if (!r.length) r = Array.from(g.querySelectorAll('.thumbnail-item'));
            if (!r.length) r = Array.from(g.querySelectorAll('[class*="thumbnail-item"]'));
            return r;
        }
        function posOv(item) {
            var r = item.getBoundingClientRect();
            ov.style.left = r.left + 'px'; ov.style.top = r.top + 'px';
            ov.style.width = r.width + 'px'; ov.style.height = r.height + 'px';
            ov.style.display = 'flex';
        }
        function hideOv() {
            var prevGid = curCfg ? curCfg.id : null;
            ov.style.display = 'none'; curItem = null; curCfg = null; curItemIdx = -1;
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
            var g = document.getElementById(gid); if (!g) return false;
            var items = Array.from(g.querySelectorAll('[data-testid="thumbnail"]'));
            if (!items.length) items = Array.from(g.querySelectorAll('.thumbnail-item'));
            if (!items.length) items = Array.from(g.querySelectorAll('[class*="thumbnail-item"]'));
            var item = items[idx]; if (!item) return false;
            var img = item.querySelector('img');
            if (!img || !img.src || img.src.indexOf('data:') === 0) return false;
            // 썸네일 src를 원본 src로 변환
            var origSrc = thumbSrcToOriginalSrc(img.src);
            var rawSrc = decodeURIComponent(origSrc);
            var filename = rawSrc.split('=').pop().split('/').pop().split('?')[0];
            if (!filename || !filename.match(/\.(png|jpg|jpeg|webp|gif)$/i)) filename = 'image.png';
            var a = document.createElement('a');
            a.href = origSrc;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            return true;
        }
        // input/textarea 요소에 값을 설정하고 Gradio/Svelte가 감지할 수 있는 이벤트를 발생시킴
        function setNativeValue(inp, val) {
            var proto = inp.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
            var nativeSetter = Object.getOwnPropertyDescriptor(proto, 'value');
            if (nativeSetter && nativeSetter.set) { nativeSetter.set.call(inp, val); } else { inp.value = val; }
            inp.dispatchEvent(new InputEvent('input', {bubbles: true, cancelable: true}));
            inp.dispatchEvent(new Event('change', {bubbles: true}));
        }
        // 오버레이 버튼 클릭: 라이트박스 열지 않고 서버 액션 직접 트리거
        function triggerOverlayAction(gid, key, idx) {
            var tbMap = {
                'live-gallery': {dl:'overlay-dl-gen', vid:'overlay-vid-gen', ref:'overlay-ref-gen'},
                'full-gallery': {dl:'overlay-dl-gallery', vid:'overlay-vid-gallery', ref:'overlay-ref-gallery'}
            };
            var tbId = tbMap[gid] && tbMap[gid][key]; if (!tbId) return;
            var wrapper = document.getElementById(tbId); if (!wrapper) return;
            var inp = wrapper.querySelector('input[type="text"],textarea'); if (!inp) return;
            // idx:timestamp 형식으로 설정하여 같은 인덱스 반복 클릭도 이벤트 발생
            setNativeValue(inp, idx + ':' + Date.now());
        }
        ov.addEventListener('mousedown', function(e) {
            var btn = e.target.closest('[data-k]'); if (!btn || !curItem || !curCfg) return;
            e.preventDefault(); e.stopPropagation();
            if (btn.dataset.k === 'dl') {
                // PNG 다운로드: 갤러리 img src에서 직접 다운로드 (원본 PNG 파일 서빙)
                // gallery가 file path를 사용하므로 img src = 원본 PNG 경로
                if (!downloadFromGallery(curCfg.id, curItemIdx)) {
                    // 폴백: Python 핸들러 트리거
                    triggerOverlayAction(curCfg.id, btn.dataset.k, curItemIdx);
                }
            } else {
                triggerOverlayAction(curCfg.id, btn.dataset.k, curItemIdx);
            }
        });
        window.addEventListener('scroll', function() { if (curItem && ov.style.display !== 'none') posOv(curItem); }, true);
        window.addEventListener('resize', function() { if (curItem && ov.style.display !== 'none') posOv(curItem); });
    })();

    // ── 다중 선택 (호버 시 체크박스 표시, 선택 시 항상 표시) ────────────────
    (function() {
        var GIDS = ['live-gallery', 'full-gallery'];
        var TBIDS = {'live-gallery': 'ms-state-gen', 'full-gallery': 'ms-state-gallery'};
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
        function getItems(gid) {
            var g = document.getElementById(gid); if (!g) return [];
            var r = Array.from(g.querySelectorAll('[data-testid="thumbnail"]'));
            if (!r.length) r = Array.from(g.querySelectorAll('.thumbnail-item'));
            if (!r.length) r = Array.from(g.querySelectorAll('[class*="thumbnail-item"]'));
            return r;
        }
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
        function attachObs() {
            GIDS.forEach(function(gid) {
                var g = document.getElementById(gid);
                if (g) obs.observe(g, {childList: true, subtree: true});
            });
        }
        window.addEventListener('scroll', function() { GIDS.forEach(refreshOverlays); }, true);
        window.addEventListener('resize', function() { GIDS.forEach(refreshOverlays); });
        setTimeout(attachObs, 800);
        setTimeout(function() { GIDS.forEach(refreshOverlays); }, 1500);
    })();

    // ── 라이트박스 열릴 때 썸네일 → 원본 이미지로 교체 ──────────────────────
    (function() {
        new MutationObserver(function(mutations) {
            mutations.forEach(function(m) {
                m.addedNodes.forEach(function(node) {
                    if (node.nodeType !== 1) return;
                    // Gradio 라이트박스는 fixed div에 img를 포함
                    var imgs = [];
                    if (node.tagName === 'IMG') {
                        imgs = [node];
                    } else if (typeof node.querySelectorAll === 'function') {
                        // 라이트박스 컨테이너로 추정되는 고정 위치 요소 내 이미지
                        var fixed = node.style && node.style.position === 'fixed';
                        if (fixed || node.closest && node.closest('[data-testid="lightbox"]')) {
                            imgs = Array.from(node.querySelectorAll('img'));
                        }
                        // data-testid="lightbox" 내부 이미지
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
        var DL_IDS = ['single-png-gen', 'selected-zip-gen', 'single-png-gallery', 'selected-zip-gallery', 'full-zip-gallery', 'selected-videos-zip', 'all-videos-zip'];
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
                // 파일 링크 생성 시 자동으로 다운로드 트리거
                new MutationObserver(function() {
                    var link = wrap.querySelector('a[href]');
                    if (link && link.href && !link.dataset.autoTriggered) {
                        link.dataset.autoTriggered = '1';
                        var a = document.createElement('a');
                        a.href = link.href;
                        a.download = link.download || '';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        setTimeout(function() { link.removeAttribute('data-auto-triggered'); }, 3000);
                    }
                }).observe(wrap, {childList: true, subtree: true, attributes: true, attributeFilter: ['href']});
            })();
        });
    })();
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
                            value=DEFAULT_MODEL,
                            label="모델 선택",
                            info="나노 바나나 2: 속도·대량 작업 최적화 / 나노 바나나 프로: 전문 애셋·고급 추론 / 나노 바나나: 초고속·저지연",
                        )

                        ratio_dropdown = gr.Dropdown(
                            choices=list(ASPECT_RATIOS.keys()),
                            value=DEFAULT_RATIO,
                            label="화면 비율",
                        )

                        quality_dropdown = gr.Dropdown(
                            choices=list(QUALITY_OPTIONS.keys()),
                            value=DEFAULT_QUALITY,
                            label="화질",
                        )

                        count_slider = gr.Slider(
                            minimum=1,
                            maximum=MAX_COUNT,
                            value=DEFAULT_COUNT,
                            step=1,
                            label=f"생성 개수 (최대 {MAX_COUNT}장)",
                        )

                        ref_image_upload = gr.File(
                            label=f"레퍼런스 이미지 (최대 {MAX_REFERENCE_IMAGES}장)",
                            file_count="multiple",
                            file_types=["image"],
                            elem_id="ref-upload-main",
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
                            columns=2,
                            height=300,
                            object_fit="contain",
                            visible=False,
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 프롬프트")

                        prompt_input = gr.Textbox(
                            label="프롬프트",
                            placeholder="생성할 이미지를 설명하세요. 한국어/영어 모두 사용 가능합니다.\n예: A cinematic key visual of a futuristic city at golden hour, dramatic lighting, 8K\n\n💡 Ctrl+Enter로 생성 시작",
                            lines=6,
                            elem_id="image-prompt",
                        )

                        model_info = gr.Markdown(
                            f"**{DEFAULT_MODEL}**: {MODELS[DEFAULT_MODEL]['description']}"
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
                live_gallery = gr.Gallery(
                    label="생성된 이미지",
                    columns=4,
                    height=1050,
                    object_fit="contain",
                    value=gallery_state.to_gradio_gallery(),
                    elem_id="live-gallery",
                )
                with gr.Row():
                    btn_refresh_gen = gr.Button("🔄 새로고침", variant="secondary", scale=1)
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

                # 상세보기 레퍼런스 이미지 패널
                with gr.Group(visible=False) as detail_panel_gen:
                    gr.Markdown("#### 🖼️ 이 이미지 생성에 사용된 레퍼런스")
                    detail_ref_gallery_gen = gr.Gallery(
                        label="레퍼런스 이미지",
                        columns=4,
                        height=220,
                        object_fit="contain",
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

                model_radio.change(update_model_info, inputs=[model_radio], outputs=[model_info])
                ref_image_upload.change(
                    update_ref_visibility,
                    inputs=[ref_image_upload],
                    outputs=[ref_preview],
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
                    concurrency_limit=None,  # 여러 생성 작업 동시 큐 허용
                ).then(
                    update_ref_visibility,
                    inputs=[ref_image_upload],
                    outputs=[ref_preview],
                )

                def _load_ref_detail_images(item):
                    """GalleryItem에서 레퍼런스 이미지를 미리보기 크기로 로드합니다."""
                    if item is None or not item.reference_image_paths:
                        return [], False
                    imgs = []
                    for rp in item.reference_image_paths:
                        try:
                            if os.path.exists(rp):
                                img = Image.open(rp).convert("RGB")
                                img.thumbnail((_REF_THUMBNAIL_SIZE, _REF_THUMBNAIL_SIZE), Image.LANCZOS)
                                imgs.append(img)
                        except Exception:
                            continue
                    return imgs, bool(imgs)

                def on_gen_gallery_select(evt: gr.SelectData):
                    idx = evt.index
                    item = gallery_state.get_success_item_by_visual_index(idx)
                    ref_imgs, has_refs = _load_ref_detail_images(item)
                    return idx, ref_imgs, gr.update(visible=has_refs)

                live_gallery.select(
                    on_gen_gallery_select,
                    outputs=[selected_img_idx_gen, detail_ref_gallery_gen, detail_panel_gen],
                )

                def refresh_live_gallery():
                    return gallery_state.to_gradio_gallery()

                btn_refresh_gen.click(refresh_live_gallery, outputs=[live_gallery])

                smart_download_gen_fn = build_smart_download_fn(gallery_state)
                btn_download_single_gen.click(
                    smart_download_gen_fn,
                    inputs=[ms_state_gen, selected_img_idx_gen],
                    outputs=[single_png_output_gen],
                )

                btn_use_as_ref_gen.click(
                    _use_as_ref,
                    inputs=[selected_img_idx_gen, ref_image_upload],
                    outputs=[ref_image_upload],
                )

                delete_gen_fn = build_delete_selected_fn(gallery_state)
                btn_delete_gen.click(
                    delete_gen_fn,
                    inputs=[ms_state_gen, selected_img_idx_gen],
                    outputs=[live_gallery, gen_status, ms_state_gen, selected_img_idx_gen],
                )

                # 오버레이 액션 핸들러 연결
                def _parse_overlay_idx(val: str) -> int:
                    """'idx:timestamp' 형식 또는 단순 숫자에서 인덱스 추출"""
                    try:
                        return int(val.split(":")[0])
                    except Exception:
                        return -1

                download_single_gen_fn = build_download_single_fn(gallery_state)
                overlay_dl_gen.input(
                    lambda val: download_single_gen_fn(_parse_overlay_idx(val)),
                    inputs=[overlay_dl_gen],
                    outputs=[single_png_output_gen],
                )
                overlay_ref_gen.input(
                    lambda val, files: _use_as_ref(_parse_overlay_idx(val), files),
                    inputs=[overlay_ref_gen, ref_image_upload],
                    outputs=[ref_image_upload],
                )

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
                            value=KLING_DEFAULT_MODEL,
                            label="Kling 모델",
                            info=" | ".join(
                                f"{k}: {v['description']}"
                                for k, v in KLING_MODELS.items()
                            ),
                        )

                        kling_quality_dropdown = gr.Dropdown(
                            choices=list(KLING_QUALITY_OPTIONS.keys()),
                            value=KLING_DEFAULT_QUALITY,
                            label="화질",
                            info="720p (Standard): 빠름 / 1080p (Professional): 고화질",
                        )

                        kling_duration_slider = gr.Slider(
                            minimum=3,
                            maximum=15,
                            value=KLING_DEFAULT_DURATION,
                            step=1,
                            label="영상 길이 (초)",
                            info="3초부터 15초까지 1초 단위로 선택",
                        )

                        kling_ratio_dropdown = gr.Dropdown(
                            choices=KLING_VIDEO_RATIOS,
                            value=KLING_DEFAULT_RATIO,
                            label="화면 비율",
                        )

                        kling_prompt_input = gr.Textbox(
                            label="프롬프트 (선택)",
                            placeholder="영상 움직임을 설명하세요. 예: Camera slowly zooms in, dramatic lighting\n\n💡 Ctrl+Enter로 생성 시작",
                            lines=3,
                            elem_id="video-prompt",
                        )

                        enable_audio_checkbox = gr.Checkbox(
                            label="🔊 오디오 생성 사용 (Kling v3 / v3 Omni 지원)",
                            value=False,
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

                gr.Markdown("### 🎥 생성된 영상")
                video_output = gr.Video(label="최근 생성 결과", height=480)

                # ── 결과물 관리 창 ────────────────────────────────────────────
                gr.Markdown("### 📹 결과물 관리")
                gr.Markdown("💡 생성된 영상 목록에서 항목을 선택하여 재생하거나 ZIP으로 다운로드할 수 있습니다.")
                existing_videos = load_existing_video_outputs()
                _vid_choices = [(v["label"], v["path"]) for v in existing_videos]

                with gr.Row():
                    btn_refresh_videos = gr.Button("🔄 새로고침", variant="secondary", scale=1)
                    btn_download_selected_videos = gr.Button(
                        "📦 선택 ZIP", variant="secondary", scale=1
                    )
                    btn_download_all_videos = gr.Button(
                        "⬇️ 전체 ZIP", variant="secondary", scale=1
                    )
                video_gallery_dropdown = gr.Dropdown(
                    choices=_vid_choices,
                    label="생성된 영상 목록 (최신순) — 항목 선택 시 아래에서 재생, 여러 항목 선택 후 ZIP 다운로드 가능",
                    multiselect=True,
                    interactive=True,
                    value=None,
                )
                video_gallery_player = gr.Video(
                    label="선택한 영상 재생",
                    height=480,
                    visible=bool(_vid_choices),
                )
                selected_videos_zip_output = gr.File(label="선택 영상 ZIP 다운로드", elem_id="selected-videos-zip")
                all_videos_zip_output = gr.File(label="전체 영상 ZIP 다운로드", elem_id="all-videos-zip")

                def _on_kling_model_change(model_label: str):
                    """Omni 모델(supports_video_reference=True)에서만 영상 레퍼런스 탭 표시"""
                    supports = KLING_MODELS.get(model_label, {}).get("supports_video_reference", False)
                    return gr.update(visible=supports)

                kling_model_radio.change(
                    _on_kling_model_change,
                    inputs=[kling_model_radio],
                    outputs=[video_ref_tab],
                )

                unified_video_fn = build_unified_video_fn(gallery_state)
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
                    outputs=[video_output, video_status],
                ).then(
                    lambda: (
                        gr.update(
                            choices=[(v["label"], v["path"]) for v in load_existing_video_outputs()],
                            value=None,
                        ),
                        gr.update(visible=True),
                    ),
                    outputs=[video_gallery_dropdown, video_gallery_player],
                )

                def on_video_gallery_select(paths):
                    if not paths:
                        return gr.update(value=None, visible=False)
                    path = paths[0] if isinstance(paths, list) else paths
                    if path and _is_safe_output_path(path) and os.path.exists(path):
                        return gr.update(value=path, visible=True)
                    return gr.update(value=None, visible=False)

                video_gallery_dropdown.change(
                    on_video_gallery_select,
                    inputs=[video_gallery_dropdown],
                    outputs=[video_gallery_player],
                )

                def refresh_video_gallery():
                    videos = load_existing_video_outputs()
                    choices = [(v["label"], v["path"]) for v in videos]
                    return gr.update(choices=choices, value=None), gr.update(visible=bool(choices))

                btn_refresh_videos.click(
                    refresh_video_gallery,
                    outputs=[video_gallery_dropdown, video_gallery_player],
                )

                def download_selected_videos(paths):
                    if not paths:
                        gr.Warning("다운로드할 영상을 선택해주세요.")
                        return None
                    existing = [
                        p for p in paths
                        if p and _is_safe_output_path(p) and os.path.exists(p)
                    ]
                    if not existing:
                        gr.Warning("선택한 영상 파일을 찾을 수 없습니다.")
                        return None
                    return create_zip_from_paths(existing)

                btn_download_selected_videos.click(
                    download_selected_videos,
                    inputs=[video_gallery_dropdown],
                    outputs=[selected_videos_zip_output],
                )

                def download_all_videos():
                    videos = load_existing_video_outputs()
                    paths = [
                        v["path"] for v in videos
                        if v.get("path") and _is_safe_output_path(v["path"]) and os.path.exists(v["path"])
                    ]
                    if not paths:
                        gr.Warning("다운로드할 영상이 없습니다.")
                        return None
                    return create_zip_from_paths(paths)

                btn_download_all_videos.click(
                    download_all_videos,
                    outputs=[all_videos_zip_output],
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

                full_gallery = gr.Gallery(
                    label="전체 갤러리",
                    columns=4,
                    height=900,
                    object_fit="contain",
                    value=gallery_state.to_gradio_gallery(),
                    elem_id="full-gallery",
                )

                with gr.Row():
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
                        columns=4,
                        height=220,
                        object_fit="contain",
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
                    return idx, ref_imgs, gr.update(visible=has_refs)

                full_gallery.select(
                    on_gallery_tab_select,
                    outputs=[selected_img_idx_gallery, detail_ref_gallery_full, detail_panel_gallery],
                )

                smart_download_gallery_fn = build_smart_download_fn(gallery_state)
                btn_download_single_gallery.click(
                    smart_download_gallery_fn,
                    inputs=[ms_state_gallery, selected_img_idx_gallery],
                    outputs=[single_png_output_gallery],
                )

                btn_use_as_ref_gallery.click(
                    _use_as_ref,
                    inputs=[selected_img_idx_gallery, ref_image_upload],
                    outputs=[ref_image_upload],
                )

                delete_gallery_fn = build_delete_selected_fn(gallery_state)
                btn_delete_gallery.click(
                    delete_gallery_fn,
                    inputs=[ms_state_gallery, selected_img_idx_gallery],
                    outputs=[full_gallery, gallery_status, ms_state_gallery, selected_img_idx_gallery],
                )

                download_single_gallery_fn = build_download_single_fn(gallery_state)
                overlay_dl_gallery.input(
                    lambda val: download_single_gallery_fn(_parse_overlay_idx(val)),
                    inputs=[overlay_dl_gallery],
                    outputs=[single_png_output_gallery],
                )
                overlay_ref_gallery.input(
                    lambda val, files: _use_as_ref(_parse_overlay_idx(val), files),
                    inputs=[overlay_ref_gallery, ref_image_upload],
                    outputs=[ref_image_upload],
                )

        # ── 탭 간 "영상화" 버튼 공통 핸들러 ─────────────────────────────────

        def on_make_video(idx: int):
            """선택된 이미지를 영상 생성 탭으로 전송합니다."""
            item = gallery_state.get_success_item_by_visual_index(idx)
            if item is None:
                gr.Warning("이미지를 먼저 클릭하여 선택해주세요.")
                return None, gr.update()
            # 원본 이미지를 디스크에서 로드 (item.image는 썸네일일 수 있음)
            try:
                img = Image.open(item.image_path).convert("RGB") if item.image_path and os.path.exists(item.image_path) else item.image
            except Exception:
                img = item.image
            return img, gr.update(selected="tab_video")

        btn_make_video_gen.click(
            on_make_video,
            inputs=[selected_img_idx_gen],
            outputs=[ref_image_vid, main_tabs],
        )

        btn_make_video_gallery.click(
            on_make_video,
            inputs=[selected_img_idx_gallery],
            outputs=[ref_image_vid, main_tabs],
        )

        overlay_vid_gen.input(
            lambda val: on_make_video(_parse_overlay_idx(val)),
            inputs=[overlay_vid_gen],
            outputs=[ref_image_vid, main_tabs],
        )

        overlay_vid_gallery.input(
            lambda val: on_make_video(_parse_overlay_idx(val)),
            inputs=[overlay_vid_gallery],
            outputs=[ref_image_vid, main_tabs],
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
