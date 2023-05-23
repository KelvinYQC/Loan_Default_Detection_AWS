FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data/subsample_train.csv ./data/subsample_train.csv
COPY src ./src
COPY pipeline.py .
COPY config ./config

CMD ["python3", "pipeline.py"]