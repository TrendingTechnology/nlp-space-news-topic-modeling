FROM python:3.8.6-slim-buster

WORKDIR /app

COPY ./app /app
# RUN ls -la /app/*

RUN pip install -r requirements.txt
# RUN pip freeze

EXPOSE 80

ENTRYPOINT [ "python3" ]
CMD ["main.py"]
