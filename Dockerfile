FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py threat_agent.py threat_models.joblib cyber_threat_logs.csv ./
ENV GOOGLE_GENAI_USE_ENTERPRISE=True GOOGLE_CLOUD_LOCATION=global
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]

