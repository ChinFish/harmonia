FROM tensorflow/tensorflow

RUN pip install --upgrade pip && \
    pip install --upgrade protobuf \
    pip install grpcio grpcio-tools

ADD . /app
ADD ./train /train
ENTRYPOINT [ "python", "/app/main.py" ]