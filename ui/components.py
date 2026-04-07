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
)
from core.gemini_client import validate_api_key, generate_batch_images
from core.image_utils import (
    load_reference_images,
    save_image,
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
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key and key != "your_google_api_key_here":
                    return {"api_key": key}
    return {}


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
            paths = [r if isinstance(r, str) else r.name for r in ref_images if r is not None]
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
        return gallery_state.to_gradio_gallery(), gallery_state.get_summary(), None

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
            <div class="subtitle-text">나노 바나나 2 / 나노 바나나 프로 모델로 영상 키 이미지를 생성합니다.</div>
            """
        )

        with gr.Tabs():

            # ── 탭 1: API 키 설정 ────────────────────────────────────────────
            with gr.Tab("🔑 API 키 설정"):
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

            # ── 탭 2: 이미지 생성 ────────────────────────────────────────────
            with gr.Tab("🖼️ 이미지 생성"):
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
                        )

                # 결과 갤러리 (생성 탭 하단)
                gr.Markdown("### 🖼️ 생성 결과")
                live_gallery = gr.Gallery(
                    label="생성된 이미지",
                    columns=3,
                    height=500,
                    object_fit="contain",
                )

                # 모델 선택 변경 시 설명 업데이트
                def update_model_info(m):
                    return f"**{m}**: {MODELS[m]['description']}"

                def update_ref_visibility(files):
                    if not files:
                        return gr.Gallery(visible=False, value=[])
                    imgs = []
                    for f in files:
                        p = f if isinstance(f, str) else f.name
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

            # ── 탭 3: 갤러리 & 다운로드 ──────────────────────────────────────
            with gr.Tab("📁 갤러리 & 다운로드"):
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

                zip_file_output = gr.File(label="ZIP 다운로드", visible=True)

                def refresh_gallery():
                    return gallery_state.to_gradio_gallery(), gallery_state.get_summary()

                btn_refresh.click(refresh_gallery, outputs=[full_gallery, gallery_status])

                download_zip_fn = build_download_zip_fn(gallery_state)
                btn_download_zip.click(download_zip_fn, outputs=[zip_file_output])

                clear_fn = build_clear_fn(gallery_state)
                btn_clear.click(clear_fn, outputs=[full_gallery, gallery_status, zip_file_output])

        gr.Markdown(
            """
            ---
            **안내**: 생성된 이미지는 `outputs/YYYY-MM-DD/` 폴더에 자동 저장됩니다.
            나노 바나나 2와 나노 바나나 프로 모두 레퍼런스 이미지를 지원합니다.
            """
        )

    return demo
