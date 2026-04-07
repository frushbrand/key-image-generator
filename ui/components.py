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
    create_image_to_video_task,
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


def build_generate_video_fn(gallery_state: GalleryState):
    """영상 생성 버튼 클릭 핸들러 팩토리"""

    def generate_video(
        access_key: str,
        secret_key: str,
        ref_image,       # gr.Image → PIL Image or None
        prompt: str,
        model_label: str,
        duration: int,
        aspect_ratio: str,
        progress=gr.Progress(track_tqdm=True),
    ):
        ak = (access_key or "").strip()
        sk = (secret_key or "").strip()
        if not ak or not sk:
            gr.Warning("Kling Access Key와 Secret Key를 입력해주세요.")
            return None, "❌ API 키 없음"

        # 레퍼런스 이미지 결정
        pil_image: Optional[Image.Image] = None
        if ref_image is not None:
            if isinstance(ref_image, Image.Image):
                pil_image = ref_image
            else:
                pil_image = Image.fromarray(ref_image).convert("RGB")

        if pil_image is None:
            gr.Warning("레퍼런스 이미지를 업로드하거나 갤러리에서 '영상화' 버튼을 클릭해주세요.")
            return None, "❌ 레퍼런스 이미지 없음"

        model_cfg = KLING_MODELS[model_label]
        api_model = model_cfg["api_name"]
        mode = model_cfg["mode"]

        progress(0.05, desc="영상 생성 작업 요청 중...")
        try:
            task_id = create_image_to_video_task(
                access_key=ak,
                secret_key=sk,
                image=pil_image,
                prompt=prompt or "",
                model=api_model,
                duration=int(duration),
                aspect_ratio=aspect_ratio,
                mode=mode,
            )
        except Exception as e:
            return None, f"❌ 작업 생성 실패: {e}"

        progress(0.15, desc=f"대기 중... (task_id: {task_id[:8]}…)")

        video_url = None

        def on_poll(elapsed, status):
            frac = min(0.15 + elapsed / 300 * 0.80, 0.95)
            progress(frac, desc=f"처리 중… {elapsed}초 경과 (상태: {status})")

        try:
            video_url = poll_task_result(
                access_key=ak,
                secret_key=sk,
                task_id=task_id,
                timeout=300,
                poll_interval=5,
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
        api_key: str,
        model_name: str,
        prompt: str,
        ratio: str,
        quality: str,
        count: int,
        ref_images,  # Gradio File 컴포넌트 값 (list of paths or None)
        progress=gr.Progress(track_tqdm=True),
    ):
        if not api_key or not api_key.strip():
            gr.Warning("API 키를 입력해주세요.")
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

    # 탭 전환을 위한 JavaScript (localStorage로 새로고침 시 탭 유지)
    TAB_PERSIST_JS = """
    <script>
    (function() {
        var TAB_KEY = 'keyImageGenTab';

        function saveTab(idx) {
            try { localStorage.setItem(TAB_KEY, String(idx)); } catch(e) {}
        }

        function restoreTab() {
            try {
                var idx = localStorage.getItem(TAB_KEY);
                if (idx === null) return;
                var attempt = 0;
                // Poll up to 40 times (4 seconds) waiting for Gradio to render tab buttons
                var iv = setInterval(function() {
                    var buttons = document.querySelectorAll('.tab-nav button');
                    if (buttons.length > parseInt(idx)) {
                        buttons[parseInt(idx)].click();
                        clearInterval(iv);
                    }
                    if (++attempt > 40) clearInterval(iv);
                }, 100);
            } catch(e) {}
        }

        document.addEventListener('click', function(e) {
            var btn = e.target.closest('.tab-nav button');
            if (btn) {
                var buttons = Array.from(document.querySelectorAll('.tab-nav button'));
                var idx = buttons.indexOf(btn);
                if (idx >= 0) saveTab(idx);
            }
        }, true);

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                // 300ms delay gives Gradio time to finish rendering the tab UI
                setTimeout(restoreTab, 300);
            });
        } else {
            // 300ms delay gives Gradio time to finish rendering the tab UI
            setTimeout(restoreTab, 300);
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

                        api_key_gen = gr.Textbox(
                            label="Google API 키",
                            placeholder="AIza... (API 키 설정 탭에서 저장하면 자동 입력)",
                            value=saved_api_key,
                            type="password",
                        )

                        model_radio = gr.Radio(
                            choices=list(MODELS.keys()),
                            value=DEFAULT_MODEL,
                            label="모델 선택",
                            info="나노 바나나 2: 빠름 / 나노 바나나 프로: 고품질",
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
                gr.Markdown("💡 이미지를 클릭해서 선택한 뒤 **🎬 영상화** 버튼을 눌러 영상을 생성할 수 있습니다.")
                live_gallery = gr.Gallery(
                    label="생성된 이미지",
                    columns=3,
                    height=500,
                    object_fit="contain",
                )
                with gr.Row():
                    gen_selected_info = gr.Textbox(
                        label="선택된 이미지",
                        value="이미지를 클릭하여 선택하세요",
                        interactive=False,
                        scale=3,
                    )
                    btn_make_video_gen = gr.Button(
                        "🎬 영상화",
                        variant="secondary",
                        scale=1,
                    )

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
                        api_key_gen,
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

            # ── 탭 3: 영상 생성 (Kling) ──────────────────────────────────────
            with gr.Tab("🎬 영상 생성 (Kling)", id="tab_video"):
                gr.Markdown(
                    """
                    ### 🎬 Kling AI 영상 생성
                    이미지 생성 탭의 갤러리에서 **영상화** 버튼을 클릭하거나, 아래에 이미지를 직접 업로드하세요.
                    Kling API 키는 **API 키 설정** 탭에서 저장할 수 있습니다.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🔑 Kling API 키")
                        kling_access_vid = gr.Textbox(
                            label="Access Key",
                            placeholder="Access Key...",
                            value=saved_kling_access,
                            type="password",
                        )
                        kling_secret_vid = gr.Textbox(
                            label="Secret Key",
                            placeholder="Secret Key...",
                            value=saved_kling_secret,
                            type="password",
                        )

                        gr.Markdown("#### 🖼️ 레퍼런스 이미지")
                        ref_image_vid = gr.Image(
                            label="레퍼런스 이미지 (갤러리에서 전송되거나 직접 업로드)",
                            type="pil",
                            height=220,
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
                            maximum=10,
                            value=KLING_DEFAULT_DURATION,
                            step=1,
                            label="영상 길이 (초)",
                            info="3초부터 10초까지 1초 단위로 선택",
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

                generate_video_fn = build_generate_video_fn(gallery_state)
                btn_generate_video.click(
                    generate_video_fn,
                    inputs=[
                        kling_access_vid,
                        kling_secret_vid,
                        ref_image_vid,
                        kling_prompt_input,
                        kling_model_radio,
                        kling_duration_slider,
                        kling_ratio_dropdown,
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
                    btn_make_video_gallery = gr.Button(
                        "🎬 영상화",
                        variant="secondary",
                        scale=1,
                    )

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
