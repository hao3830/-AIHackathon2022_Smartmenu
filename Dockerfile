FROM python:3.7

RUN apt-get update && apt-get install ffmpeg -y && apt-get clean

RUN pip install --upgrade pip

RUN pip install googletrans==4.0.0-rc1 \
    torch==1.11.0 \
    torchvision==0.12.0 \
    python-Levenshtein \
    fuzzywuzzy==0.18.0 \
    Flask==2.1.2 \
    onnxruntime \
    symspellpy \
    easydict \
    gdown \
    opencv-python \
    pyyaml \
    unidecode \
    shapely \
    vietocr

# RUN pip install easyocr

# Clean up pip cache
RUN rm -rf ~/.cache/pip

WORKDIR /app
# Download weights
RUN mkdir -p weights \
    # && gdown -O weights/yolov5.onnx 11sk_VWNY8BGAeMnoA1x0oFzdidO3roik \
    # && gdown -O weights/encoder.onnx 1eScywo-d_xBfYl_6dBgLLtHZGoltoqCd \
    # && gdown -O weights/decoder.onnx 1Qr00YCHocPe2GgpK56KbUsUljd9vHFaR \
    # && gdown -O weights/cnn.onnx 18ajwQsGCRl9w7JWgzvdOZjhHFZCyfEyf \
    && gdown -O weights/yolov5.pt 1tjiXzrQtvd1qt_7XDu8-sToM_yG80_Jz

COPY . /app

RUN python3 warmup.py

EXPOSE 5000

CMD ["python", "./main.py"]
