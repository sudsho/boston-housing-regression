FROM python:3.7-slim

# build tools occasionally needed by numpy/scipy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first so docker layer cache is friendly
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# bake a trained model into the image so the api works on first start
RUN python -m src.train --config configs/default.yaml

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "2", "--log-file=-", "app:app"]
