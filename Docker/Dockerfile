FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY ["Too lazy to care abt/requirements.txt", "./requirements.txt"]

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir docling

COPY . .

EXPOSE 8899

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8899", "--no-browser", "--allow-root"]
