FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Streamlit 포트
EXPOSE 8501

# 컨테이너 내부 헬스체크 (EB가 참고)
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 앱 실행
ENTRYPOINT ["streamlit","run","chat.py","--server.port=8501","--server.address=0.0.0.0"]
