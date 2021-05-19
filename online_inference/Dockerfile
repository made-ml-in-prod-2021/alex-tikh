FROM python:3.8-alpine
COPY requirements.txt ./requirements.txt
RUN apk update
RUN apk --no-cache add curl gcc g++
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src /src
COPY app.py /app.py
COPY models /models

WORKDIR .

ENV PATH_TO_MODEL="models/logistic_regression.pkl"
ENV PATH_TO_TRANSFORMER="models/transformer.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]