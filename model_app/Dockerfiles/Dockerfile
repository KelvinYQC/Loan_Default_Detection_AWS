FROM --platform=linux/x86_64 python:3.10-slim

WORKDIR /app

RUN /usr/local/bin/python -m pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY config ./config
COPY src ./src

EXPOSE 80

CMD ["streamlit", "run", "--server.port=80", "--server.address=0.0.0.0", "--server.fileWatcherType=none", "src/app.py"]