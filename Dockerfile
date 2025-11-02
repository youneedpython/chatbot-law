FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Streamlit
EXPOSE 8501
ENTRYPOINT ["streamlit","run","chat.py","--server.port=8501","--server.address=0.0.0.0"]
