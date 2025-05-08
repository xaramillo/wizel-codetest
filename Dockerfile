FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_model.py .
COPY predict.py .

VOLUME /data
VOLUME /models

CMD ["bash"]