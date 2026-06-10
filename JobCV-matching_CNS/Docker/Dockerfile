FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.api.txt requirements.experiments.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8899

CMD ["python", "-m", "notebook", "--ip=0.0.0.0", "--port=8899", "--no-browser", "--allow-root"]
