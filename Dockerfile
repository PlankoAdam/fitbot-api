FROM python:3.11

WORKDIR /app/

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
EXPOSE 5000

COPY ./app /app

CMD [ "python", "/app/main.py" ]