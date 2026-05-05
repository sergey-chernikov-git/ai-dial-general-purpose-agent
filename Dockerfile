FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN apt update
RUN apt install tcpdump  -y
COPY task /app/task

CMD ["python", "/app/task/app.py"]