FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV NAME HousePricePrediction

CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
