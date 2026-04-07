FROM python:3.11-slim

WORKDIR /app

# 의존성 먼저 복사 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# outputs 폴더 생성
RUN mkdir -p outputs

# 7860 포트 노출
EXPOSE 7860

CMD ["python", "app.py"]
