FROM python:3.14.10-slim-bookworm

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "xGboost_model.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["python", "-m","waitress", "--listen=0.0.0.0:9696", "predict:app"]
