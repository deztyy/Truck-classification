FROM python:3.9-slim-bullseye
WORKDIR /app
RUN pip install redis numpy
COPY main.py .
CMD ["python", "main.py"]