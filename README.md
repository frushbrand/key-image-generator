# 🎨 AI 영상 키 이미지 생성 툴

Google Gemini API와 Kling AI API를 활용한 로컬 실행형 키 이미지 + 영상 생성 도구입니다.  
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

## 🚀 빠른 시작

### 사전 요구사항
- Python 3.10 이상
- Google Gemini API 키 ([AI Studio에서 발급](https://aistudio.google.com/app/apikey))
- Kling AI API 키 ([Kling 개발자 콘솔에서 발급](https://kling.ai/dev/api-key)) ← 영상 생성 기능 사용 시

---

### 🟢 방법 1 — 원클릭 설치 스크립트 (비개발자 권장)

Python만 설치되어 있으면 스크립트 하나로 자동 설치 후 실행됩니다.

#### Windows
1. [Python 3.10+](https://www.python.org/downloads/) 설치 (설치 시 **"Add Python to PATH"** 옵션 반드시 체크)
2. 이 저장소를 [ZIP으로 다운로드](https://github.com/frushbrand/key-image-generator/archive/refs/heads/main.zip) 후 압축 해제
3. 압축 해제된 폴더에서 **`setup.bat`** 파일을 **더블클릭**
4. 브라우저가 자동으로 열리며 **http://localhost:7860** 접속

#### macOS / Linux
```bash
# 저장소 클론 후
chmod +x setup.sh
./setup.sh
```

> 이후 실행 시에도 같은 스크립트를 실행하면 됩니다. (패키지 재설치 없이 빠르게 시작)

---

### 🐳 방법 2 — Docker (환경 문제 없이 가장 안정적)

[Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치 후:

```bash
# 저장소 클론
git clone https://github.com/frushbrand/key-image-generator.git
cd key-image-generator

# 실행 (최초 1회 빌드 후 바로 시작)
docker compose up --build
```

브라우저에서 **http://localhost:7860** 접속  
종료: `Ctrl+C` / 이후 재실행: `docker compose up`

---

### ⚙️ 방법 3 — 직접 설치 (개발자용)

> **명령어 입력 위치**: macOS/Linux는 **터미널**, Windows는 **명령 프롬프트(cmd)** 또는 **PowerShell**을 열고 아래 명령을 순서대로 입력하면 됩니다.

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

### Google Gemini API 키

**방법 1 – UI에서 입력 (권장)**  
앱 실행 후 `🔑 API 키 설정` 탭에서 키를 입력하고 저장하면 `.app_settings.json`에 보관됩니다.

**방법 2 – .env 파일**  
```bash
cp .env.example .env
# .env 파일을 열어 GOOGLE_API_KEY=여기에_키_입력
```

### Kling AI API 키 (영상 생성)

1. [https://kling.ai/dev/api-key](https://kling.ai/dev/api-key) 접속 후 로그인
2. **API 키 생성** → **Access Key**와 **Secret Key** 복사 (Secret Key는 최초 1회만 표시)
3. 앱의 `🔑 API 키 설정` 탭 하단 Kling 섹션에 입력 후 저장

---

## 📁 디렉토리 구조

```
key-image-generator/
├── app.py                  # 메인 진입점
├── requirements.txt        # Python 의존성
├── .env.example            # 환경변수 예시
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

## ⚠️ 주의사항

- Gemini 3 Pro(나노 바나나 프로)는 Preview 모델로 접근 권한이 필요할 수 있습니다.
- API 요청 한도(quota)에 따라 생성이 제한될 수 있습니다.
- 생성된 이미지와 영상은 로컬에만 저장되며, 외부로 전송되지 않습니다.
- Kling API는 유료 서비스입니다. 사용 전 Kling 요금 정책을 확인하세요.

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