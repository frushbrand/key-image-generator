# 🎨 AI 영상 키 이미지 생성 툴

Google Gemini API를 활용한 로컬 실행형 키 이미지 생성 도구입니다.  
**나노 바나나 2** (Gemini 2.0 Flash Image) 및 **나노 바나나 프로** (Imagen 3) 모델을 지원합니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 🤖 모델 선택 | 나노 바나나 2 (빠름) / 나노 바나나 프로 (고품질) |
| 📐 화면 비율 | 1:1 · 16:9 · 9:16 · 4:3 · 3:4 · 2:3 · 3:2 |
| 🖼️ 화질 설정 | Standard / HD / Ultra HD |
| 🗂️ 레퍼런스 이미지 | 최대 4장 업로드 (나노 바나나 2 전용) |
| 🔢 배치 생성 | 1~20장 동시 병렬 생성 |
| 💾 자동 저장 | `outputs/YYYY-MM-DD/` 폴더에 PNG + JSON 메타데이터 |
| 📦 ZIP 다운로드 | 생성된 이미지 일괄 다운로드 |
| 🔑 API 키 영속화 | 로컬 파일에 저장, 재실행 시 자동 로드 |

---

## 🚀 빠른 시작

### 사전 요구사항
- Python 3.10 이상
- Google Gemini API 키 ([AI Studio에서 발급](https://aistudio.google.com/app/apikey))

### 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/frushbrand/key-image-generator.git
cd key-image-generator

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 실행
python app.py
```

브라우저에서 **http://localhost:7860** 접속

---

## 🔑 API 키 설정 방법

**방법 1 – UI에서 입력 (권장)**  
앱 실행 후 `🔑 API 키 설정` 탭에서 키를 입력하고 저장하면 `.app_settings.json`에 보관됩니다.

**방법 2 – .env 파일**  
```bash
cp .env.example .env
# .env 파일을 열어 GOOGLE_API_KEY=여기에_키_입력
```

---

## 📁 디렉토리 구조

```
key-image-generator/
├── app.py                  # 메인 진입점
├── requirements.txt        # Python 의존성
├── .env.example            # 환경변수 예시
├── config/
│   └── settings.py         # 모델·화질·비율 상수
├── core/
│   ├── gemini_client.py    # Gemini API 연동
│   └── image_utils.py      # 이미지 처리·저장
├── ui/
│   ├── components.py       # Gradio UI 전체
│   └── gallery.py          # 갤러리 상태 관리
└── outputs/                # 생성 이미지 자동 저장
    └── YYYY-MM-DD/
        ├── *.png
        └── *.json          # 메타데이터 (프롬프트·설정)
```

---

## 📝 사용 방법

1. **API 키 설정** 탭에서 Google API 키를 입력·저장합니다.
2. **이미지 생성** 탭으로 이동합니다.
3. 모델, 비율, 화질, 생성 개수를 선택합니다.
4. 필요한 경우 레퍼런스 이미지를 업로드합니다 (나노 바나나 2만 지원).
5. 프롬프트를 입력하고 **🚀 이미지 생성** 버튼을 클릭합니다.
6. **갤러리 & 다운로드** 탭에서 전체 결과를 확인하고 ZIP으로 다운로드합니다.

---

## ⚠️ 주의사항

- Imagen 3(나노 바나나 프로)는 Vertex AI 또는 Gemini API 접근 권한이 필요합니다.
- API 요청 한도(quota)에 따라 생성이 제한될 수 있습니다.
- 생성된 이미지는 로컬에만 저장되며, 외부로 전송되지 않습니다.

---

## 🛠️ 모델 정보

| 이름 | API 모델명 | 특징 |
|------|-----------|------|
| 나노 바나나 2 | `gemini-2.0-flash-preview-image-generation` | 빠른 생성, 레퍼런스 이미지 지원 |
| 나노 바나나 프로 | `imagen-3.0-generate-002` | 고품질, 포토리얼리스틱 |