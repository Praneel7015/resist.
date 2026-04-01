FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY inference/ ./inference/
COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py .

CMD ["app.handler"]