FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/src
COPY ../src /app/src

EXPOSE 8501

CMD ["streamlit", "run", "src/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
