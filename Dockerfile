FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN curl -L "https://YOUR_BUCKET.s3.YOUR_REGION.amazonaws.com/band_detector.onnx" \
    -o inference/models/band_detector.onnx

COPY inference/ ./inference/
COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py .

CMD ["app.handler"]