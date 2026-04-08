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
    KLING_MODELS,
    KLING_DEFAULT_DURATION,
    KLING_DEFAULT_MODEL,
    KLING_VIDEO_RATIOS,
    KLING_DEFAULT_RATIO,
)
from core.gemini_client import validate_api_key, generate_batch_images
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
)
from ui.gallery import GalleryState, GalleryItem


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
        mode_api = model_cfg["mode"]

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

        def on_poll(elapsed, status):
            frac = min(0.15 + elapsed / 300 * 0.80, 0.95)
            progress(frac, desc=f"처리 중… {elapsed}초 경과 (상태: {status})")

        try:
            video_url = poll_task_result(
                access_key=ak, secret_key=sk,
                task_id=task_id, endpoint=endpoint,
                timeout=300, poll_interval=5,
                progress_callback=on_poll,
            )
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
    """생성 버튼 클릭 핸들러 팩토리"""

    def generate(
        model_name: str,
        prompt: str,
        ratio: str,
        quality: str,
        count: int,
        ref_images,          # Gradio File 컴포넌트 값 (list of paths or None)
        progress=gr.Progress(track_tqdm=True),
    ):
        cfg = load_settings()
        api_key = cfg.get("api_key", "")
        if not api_key or not api_key.strip():
            gr.Warning("API 키 설정 탭에서 Google API 키를 먼저 저장해주세요.")
            return gallery_state.to_gradio_gallery(), gallery_state.get_summary(), None

        if not prompt or not prompt.strip():
            gr.Warning("프롬프트를 입력해주세요.")
            return gallery_state.to_gradio_gallery(), gallery_state.get_summary(), None

        count = max(1, min(int(count), MAX_COUNT))
        progress(0, desc="준비 중...")

        # 레퍼런스 이미지 로드
        ref_pil_images: list[Image.Image] = []
        if ref_images:
            paths = [r if isinstance(r, str) else r.path for r in ref_images if r is not None]
            ref_pil_images = load_reference_images(paths[:MAX_REFERENCE_IMAGES])

        completed = [0]
        new_items: list[GalleryItem] = []

        def on_progress(idx, img, err):
            completed[0] += 1
            progress(completed[0] / count, desc=f"생성 중... {completed[0]}/{count}")
            offset = len(gallery_state.items)
            if img is not None:
                img_path, _ = save_image(img, model_name, ratio, prompt, quality)
                item = GalleryItem(
                    image=img,
                    image_path=img_path,
                    model=model_name,
                    ratio=ratio,
                    quality=quality,
                    prompt=prompt,
                    index=offset + idx,
                    status="success",
                )
            else:
                item = GalleryItem(
                    image=None,
                    image_path="",
                    model=model_name,
                    ratio=ratio,
                    quality=quality,
                    prompt=prompt,
                    index=offset + idx,
                    status="failed",
                    error=err,
                )
            new_items.append(item)

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

        # 갤러리 상태에 추가 (index 순 정렬)
        new_items.sort(key=lambda x: x.index)
        for item in new_items:
            gallery_state.add(item)

        progress(1.0, desc="완료!")

        # 실패 항목 오류 메시지 수집
        failed_items = [item for item in new_items if item.status == "failed"]
        status_msg = gallery_state.get_summary()
        if failed_items:
            error_details = "\n".join(
                f"  • #{item.index + 1}: {item.error}" for item in failed_items
            )
            status_msg += f"\n\n❌ 실패 원인:\n{error_details}"

        return gallery_state.to_gradio_gallery(), status_msg, None

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
        success_items = [i for i in gallery_state.items if i.status == "success"]
        if not (0 <= idx < len(success_items)):
            gr.Warning("이미지를 먼저 클릭하여 선택해주세요.")
            return None
        item = success_items[idx]
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


# ── UI 빌더 ──────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    gallery_state = GalleryState()
    saved = load_settings()
    saved_api_key = saved.get("api_key", "")
    saved_kling_access = saved.get("kling_access_key", "")
    saved_kling_secret = saved.get("kling_secret_key", "")

    def _use_as_ref(idx: int, current_files: list):
        """선택된 이미지를 레퍼런스 이미지 업로드 칸에 추가하는 공용 핸들러."""
        success_items = [i for i in gallery_state.items if i.status == "success"]
        if not (0 <= idx < len(success_items)):
            gr.Warning("이미지를 먼저 클릭하여 선택해주세요.")
            return gr.update()
        item = success_items[idx]

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
    (function() {
        var TAB_KEY = 'keyImageGenTab';

        function saveTab(idx) {
            try { localStorage.setItem(TAB_KEY, String(idx)); } catch(e) {}
        }

        function getTabButtons() {
            // Gradio 5: role="tab" 속성 사용
            var byRole = document.querySelectorAll('[role="tab"]');
            if (byRole.length > 0) return Array.from(byRole);
            // Fallback: older Gradio class
            return Array.from(document.querySelectorAll('.tab-nav button'));
        }

        function restoreTab() {
            try {
                var idx = localStorage.getItem(TAB_KEY);
                if (idx === null) return;
                var attempt = 0;
                // Poll up to 60 times (6 seconds) waiting for Gradio to render tab buttons
                var iv = setInterval(function() {
                    var buttons = getTabButtons();
                    if (buttons.length > parseInt(idx)) {
                        buttons[parseInt(idx)].click();
                        clearInterval(iv);
                    }
                    if (++attempt > 60) clearInterval(iv);
                }, 100);
            } catch(e) {}
        }

        document.addEventListener('click', function(e) {
            var btn = e.target.closest('[role="tab"]') || e.target.closest('.tab-nav button');
            if (btn) {
                var buttons = getTabButtons();
                var idx = buttons.indexOf(btn);
                if (idx >= 0) saveTab(idx);
            }
        }, true);

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(restoreTab, 500);
            });
        } else {
            setTimeout(restoreTab, 500);
        }
    })();
    </script>
    """

    with gr.Blocks(
        title="🎨 키 이미지 생성 툴",
        theme=gr.themes.Soft(),
        css="""
        .title-text { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }
        .subtitle-text { color: #666; margin-bottom: 1rem; }
        .generate-btn { background: #4f46e5 !important; color: white !important; font-size: 1.1rem !important; }
        """,
    ) as demo:

        gr.HTML(
            """
            <div class="title-text">🎨 AI 영상 키 이미지 생성 툴</div>
            <div class="subtitle-text">나노 바나나 2 / 나노 바나나 프로 모델로 영상 키 이미지를 생성하고, Kling AI로 영상을 만들어보세요.</div>
            """
        )
        gr.HTML(TAB_PERSIST_JS)

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
                        )

                        ref_preview = gr.Gallery(
                            label="레퍼런스 미리보기",
                            columns=2,
                            height=160,
                            visible=False,
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 프롬프트")

                        prompt_input = gr.Textbox(
                            label="프롬프트",
                            placeholder="생성할 이미지를 설명하세요. 한국어/영어 모두 사용 가능합니다.\n예: A cinematic key visual of a futuristic city at golden hour, dramatic lighting, 8K",
                            lines=6,
                        )

                        model_info = gr.Markdown(
                            f"**{DEFAULT_MODEL}**: {MODELS[DEFAULT_MODEL]['description']}"
                        )

                        btn_generate = gr.Button(
                            "🚀 이미지 생성",
                            variant="primary",
                            elem_classes=["generate-btn"],
                        )

                        gen_status = gr.Textbox(
                            label="상태",
                            interactive=False,
                            value="대기 중",
                            lines=4,
                        )

                # 결과 갤러리 (생성 탭 하단)
                gr.Markdown("### 🖼️ 생성 결과")
                gr.Markdown("💡 이미지를 클릭해서 선택한 뒤 **🎬 영상 레퍼런스** 또는 **🖼️ 이미지 레퍼런스** 버튼을 눌러 활용할 수 있습니다.")
                live_gallery = gr.Gallery(
                    label="생성된 이미지",
                    columns=4,
                    height=700,
                    object_fit="contain",
                )
                with gr.Row():
                    btn_refresh_gen = gr.Button("🔄 새로고침", variant="secondary", scale=1)
                    gen_selected_info = gr.Textbox(
                        label="선택된 이미지",
                        value="이미지를 클릭하여 선택하세요",
                        interactive=False,
                        scale=3,
                    )
                    btn_download_single_gen = gr.Button(
                        "📥 PNG 다운로드",
                        variant="secondary",
                        scale=1,
                    )
                    btn_make_video_gen = gr.Button(
                        "🎬 영상 레퍼런스",
                        variant="secondary",
                        scale=1,
                    )
                    btn_use_as_ref_gen = gr.Button(
                        "🖼️ 이미지 레퍼런스",
                        variant="secondary",
                        scale=1,
                    )
                single_png_output_gen = gr.File(label="PNG 다운로드", visible=True)

                # 모델 선택 변경 시 설명 업데이트
                def update_model_info(m):
                    return f"**{m}**: {MODELS[m]['description']}"

                def update_ref_visibility(files):
                    if not files:
                        return gr.Gallery(visible=False, value=[])
                    imgs = []
                    for f in files:
                        p = f if isinstance(f, str) else f.path
                        imgs.append(p)
                    return gr.Gallery(visible=True, value=imgs)

                model_radio.change(update_model_info, inputs=[model_radio], outputs=[model_info])
                ref_image_upload.change(
                    update_ref_visibility,
                    inputs=[ref_image_upload],
                    outputs=[ref_preview],
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
                    outputs=[live_gallery, gen_status, ref_preview],
                )

                def on_gen_gallery_select(evt: gr.SelectData):
                    return evt.index, f"#{evt.index + 1}번 이미지 선택됨"

                live_gallery.select(
                    on_gen_gallery_select,
                    outputs=[selected_img_idx_gen, gen_selected_info],
                )

                def refresh_live_gallery():
                    return gallery_state.to_gradio_gallery()

                btn_refresh_gen.click(refresh_live_gallery, outputs=[live_gallery])

                download_single_gen_fn = build_download_single_fn(gallery_state)
                btn_download_single_gen.click(
                    download_single_gen_fn,
                    inputs=[selected_img_idx_gen],
                    outputs=[single_png_output_gen],
                )

                btn_use_as_ref_gen.click(
                    _use_as_ref,
                    inputs=[selected_img_idx_gen, ref_image_upload],
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

                            with gr.Tab("🎬 영상 레퍼런스"):
                                gr.Markdown("3~10초 분량의 레퍼런스 영상을 업로드하세요. 영상의 스타일·움직임을 참고하여 새로운 영상을 생성합니다.")
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
                            placeholder="영상 움직임을 설명하세요. 예: Camera slowly zooms in, dramatic lighting",
                            lines=3,
                        )

                        enable_audio_checkbox = gr.Checkbox(
                            label="🔊 오디오 생성 사용 (Kling 3 Omni 모델 전용)",
                            value=False,
                            info="Kling 3 Omni 모델에서 오디오를 함께 생성합니다. 다른 모델에서는 무시됩니다.",
                        )

                        btn_generate_video = gr.Button(
                            "🎬 영상 생성",
                            variant="primary",
                            elem_classes=["generate-btn"],
                        )

                        video_status = gr.Textbox(
                            label="상태",
                            interactive=False,
                            value="대기 중",
                        )

                gr.Markdown("#### 🎥 생성된 영상")
                video_output = gr.Video(label="생성 결과", height=480)

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
                    ],
                    outputs=[video_output, video_status],
                )

            # ── 탭 4: 갤러리 & 다운로드 ──────────────────────────────────────
            with gr.Tab("📁 갤러리 & 다운로드", id="tab_gallery"):
                gr.Markdown("### 생성된 이미지 전체 보기 및 다운로드")

                with gr.Row():
                    btn_refresh = gr.Button("🔄 갤러리 새로고침", variant="secondary")
                    btn_download_zip = gr.Button("⬇️ 전체 ZIP 다운로드", variant="primary")
                    btn_clear = gr.Button("🗑️ 갤러리 초기화", variant="stop")

                gallery_status = gr.Textbox(
                    label="상태",
                    value="아직 생성된 이미지가 없습니다.",
                    interactive=False,
                )

                full_gallery = gr.Gallery(
                    label="전체 갤러리",
                    columns=4,
                    height=600,
                    object_fit="contain",
                )

                with gr.Row():
                    gallery_selected_info = gr.Textbox(
                        label="선택된 이미지",
                        value="이미지를 클릭하여 선택하세요",
                        interactive=False,
                        scale=3,
                    )
                    btn_download_single_gallery = gr.Button(
                        "📥 PNG 다운로드",
                        variant="secondary",
                        scale=1,
                    )
                    btn_make_video_gallery = gr.Button(
                        "🎬 영상 레퍼런스",
                        variant="secondary",
                        scale=1,
                    )
                    btn_use_as_ref_gallery = gr.Button(
                        "🖼️ 이미지 레퍼런스",
                        variant="secondary",
                        scale=1,
                    )

                single_png_output_gallery = gr.File(label="PNG 다운로드", visible=True)
                zip_file_output = gr.File(label="ZIP 다운로드", visible=True)

                def refresh_gallery():
                    return gallery_state.to_gradio_gallery(), gallery_state.get_summary()

                btn_refresh.click(refresh_gallery, outputs=[full_gallery, gallery_status])

                download_zip_fn = build_download_zip_fn(gallery_state)
                btn_download_zip.click(download_zip_fn, outputs=[zip_file_output])

                clear_fn = build_clear_fn(gallery_state)
                btn_clear.click(clear_fn, outputs=[full_gallery, gallery_status, zip_file_output])

                def on_gallery_tab_select(evt: gr.SelectData):
                    return evt.index, f"#{evt.index + 1}번 이미지 선택됨"

                full_gallery.select(
                    on_gallery_tab_select,
                    outputs=[selected_img_idx_gallery, gallery_selected_info],
                )

                download_single_gallery_fn = build_download_single_fn(gallery_state)
                btn_download_single_gallery.click(
                    download_single_gallery_fn,
                    inputs=[selected_img_idx_gallery],
                    outputs=[single_png_output_gallery],
                )

                btn_use_as_ref_gallery.click(
                    _use_as_ref,
                    inputs=[selected_img_idx_gallery, ref_image_upload],
                    outputs=[ref_image_upload],
                )

        # ── 탭 간 "영상화" 버튼 공통 핸들러 ─────────────────────────────────

        def _get_image_for_video(idx: int):
            """gallery_state에서 성공한 이미지를 인덱스로 가져옵니다."""
            success_items = [i for i in gallery_state.items if i.status == "success"]
            if 0 <= idx < len(success_items):
                return success_items[idx].image
            return None

        def on_make_video(idx: int):
            """선택된 이미지를 영상 생성 탭으로 전송합니다."""
            img = _get_image_for_video(idx)
            if img is None:
                gr.Warning("이미지를 먼저 클릭하여 선택해주세요.")
                return None, gr.update()
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

        gr.Markdown(
            """
            ---
            **안내**: 생성된 이미지와 영상은 `outputs/YYYY-MM-DD/` 폴더에 자동 저장됩니다.
            나노 바나나 2와 나노 바나나 프로 모두 레퍼런스 이미지를 지원합니다.
            Kling 영상 생성은 보통 1~5분 정도 소요됩니다.
            """
        )

    return demo
