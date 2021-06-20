FROM python:3.8-slim
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY src /src
COPY app.py /app.py
COPY models /models

WORKDIR .

ENV PATH_TO_MODEL="models/logistic_regression.pkl"
ENV PATH_TO_TRANSFORMER="models/transformer.pkl"
ENV SLEEPING_TIME=20
ENV RUNNING_TIME=60

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]