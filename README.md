# 🎨 AI 영상 키 이미지 생성 툴

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/frushbrand/key-image-generator)

Google Gemini API와 Kling AI API를 활용한 키 이미지 + 영상 생성 도구입니다.  
**나노 바나나 2** (Gemini 2.0 Flash Image), **나노 바나나 프로** (Imagen 3), 그리고 **Kling 3 / 3 Pro / 3 Omni** 영상 생성을 지원합니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 🤖 이미지 모델 선택 | 나노 바나나 2 (빠름) / 나노 바나나 프로 (고품질) |
| 🎬 영상 생성 (Kling) | 생성된 이미지를 레퍼런스로 Kling 3 / Pro / Omni로 영상 생성 |
| 📐 화면 비율 | 1:1 · 16:9 · 9:16 · 4:3 · 3:4 · 2:3 · 3:2 |
| 🖼️ 화질 설정 | Standard / HD / Ultra HD |
| 🗂️ 레퍼런스 이미지 | 최대 4장 업로드 (나노 바나나 2 전용) |
| 🔢 배치 생성 | 1~20장 동시 병렬 생성 |
| 💾 자동 저장 | `outputs/YYYY-MM-DD/` 폴더에 PNG / MP4 + JSON 메타데이터 |
| 📦 ZIP 다운로드 | 생성된 이미지 일괄 다운로드 |
| 🔑 API 키 영속화 | 로컬 파일에 저장, 재실행 시 자동 로드 |

---

## 🚀 실행 방법 — GitHub Codespaces

> GitHub 계정만 있으면 아무것도 설치하지 않고 브라우저에서 바로 실행할 수 있습니다.

### 1단계 — Codespace 열기

이 페이지 상단의 **"Open in GitHub Codespaces"** 배지를 클릭하거나,  
저장소 메인 페이지에서 **Code → Codespaces → Create codespace on main** 을 선택합니다.

### 2단계 — 자동 시작 대기

Codespace가 열리면 자동으로 패키지 설치 및 앱 실행이 진행됩니다 (약 1~2분 소요).  
완료되면 포트 7860이 자동으로 포워딩되고 **앱 화면이 브라우저에 표시**됩니다.

> 앱 화면이 바로 뜨지 않으면 하단 **PORTS** 탭에서 포트 7860의 🌐 아이콘을 클릭하세요.

### 3단계 — API 키 입력

앱의 **🔑 API 키 설정** 탭에서 Google / Kling API 키를 입력하고 저장합니다.  
키는 Codespace 내부 파일(`.app_settings.json`)에 저장되며, Codespace를 재시작해도 유지됩니다.

> **보안 팁:** API 키를 영구적으로 관리하려면 저장소 설정의 **Codespaces Secrets** 에 등록하세요.  
> `GOOGLE_API_KEY`, `KLING_ACCESS_KEY`, `KLING_SECRET_KEY` 이름으로 등록하면 앱이 자동으로 불러옵니다.  
> Secrets 등록 경로: `github.com/frushbrand/key-image-generator` → **Settings → Secrets and variables → Codespaces**

---

## 🔄 Codespace 재시작 시

Codespace를 Stop 후 다시 시작하면 앱이 자동으로 재실행됩니다.  
앱이 시작되는 동안 잠깐 기다린 뒤 PORTS 탭에서 접속하면 됩니다.

앱을 수동으로 다시 시작하고 싶다면 터미널에서:

```bash
python app.py
```

---

## 🔑 API 키 발급

- **Google Gemini API 키**: [Google AI Studio](https://aistudio.google.com/app/apikey) 에서 무료 발급
- **Kling AI API 키** (영상 생성 기능): [Kling 개발자 콘솔](https://kling.ai/dev/api-key) 에서 발급 (유료)

---

## 📝 사용 방법

### 이미지 생성
1. **API 키 설정** 탭에서 Google API 키를 입력·저장합니다.
2. **이미지 생성** 탭으로 이동합니다.
3. 모델, 비율, 화질, 생성 개수를 선택합니다.
4. 필요한 경우 레퍼런스 이미지를 업로드합니다 (나노 바나나 2만 지원).
5. 프롬프트를 입력하고 **🚀 이미지 생성** 버튼을 클릭합니다.
6. **갤러리 & 다운로드** 탭에서 전체 결과를 확인하고 ZIP으로 다운로드합니다.

### Kling 영상 생성
1. **API 키 설정** 탭에서 Kling Access Key / Secret Key를 입력·저장합니다.
2. 이미지 생성 탭에서 키 이미지를 먼저 생성합니다.
3. **🎬 영상 생성 (Kling)** 탭으로 이동합니다.
4. **마지막 생성 이미지 사용** 체크 또는 이미지를 직접 업로드합니다.
5. 모델, 영상 길이, 비율을 선택하고 움직임 프롬프트를 입력합니다.
6. **🎬 영상 생성** 버튼을 클릭합니다 (보통 1~5분 소요).
7. 생성된 영상은 화면에 표시되고 `outputs/YYYY-MM-DD/` 에 자동 저장됩니다.

---

## 📁 디렉토리 구조

```
key-image-generator/
├── app.py                  # 메인 진입점
├── requirements.txt        # Python 의존성
├── .env.example            # 환경변수 예시 (Codespaces Secrets 참고용)
├── .devcontainer/
│   └── devcontainer.json   # Codespaces 환경 설정
├── config/
│   └── settings.py         # 모델·화질·비율·Kling 설정 상수
├── core/
│   ├── gemini_client.py    # Gemini API 연동 (이미지 생성)
│   ├── kling_client.py     # Kling AI 공식 API 연동 (영상 생성)
│   └── image_utils.py      # 이미지/영상 처리·저장
├── ui/
│   ├── components.py       # Gradio UI 전체
│   └── gallery.py          # 갤러리 상태 관리
└── outputs/                # 생성 이미지/영상 자동 저장
    └── YYYY-MM-DD/
        ├── *.png
        ├── *.mp4
        └── *.json          # 메타데이터 (프롬프트·설정)
```

---

## ⚠️ 주의사항

- Gemini 3 Pro(나노 바나나 프로)는 Preview 모델로 접근 권한이 필요할 수 있습니다.
- API 요청 한도(quota)에 따라 생성이 제한될 수 있습니다.
- 생성된 이미지와 영상은 Codespace 내에만 저장되며, 외부로 전송되지 않습니다.
- Kling API는 유료 서비스입니다. 사용 전 Kling 요금 정책을 확인하세요.
- Codespace는 GitHub 계정당 **월 60시간 무료** 제공됩니다. 사용 후 **Stop** 또는 **Delete** 해 두세요.

---

## 🛠️ 모델 정보

### 이미지 모델

| 이름 | API 모델명 | 특징 |
|------|-----------|------|
| 나노 바나나 2 | `gemini-flash-3.1-preview` | Gemini 3.1 Flash Image Preview, 빠른 생성, 레퍼런스 이미지 지원 |
| 나노 바나나 프로 | `gemini-3.0-pro` | Gemini 3 Pro 이미지 모델, 고품질, 레퍼런스 이미지 지원 |

### Kling 영상 모델

| 이름 | API 모델명 | 특징 |
|------|-----------|------|
| Kling 3 Standard | `kling-v3` | 빠름, 범용적 |
| Kling 3 Pro | `kling-v3-pro` | 고품질, 고해상도 |
| Kling 3 Omni | `kling-v3-omni` | 네이티브 오디오, 최고 품질 |
