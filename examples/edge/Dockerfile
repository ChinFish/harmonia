FROM pytorch/pytorch

RUN pip install --upgrade pip && \
    pip install --upgrade protobuf \
    pip install -r requirements.txt \
    pip install grpcio grpcio-tools

ADD . /app
ADD ./data /data

ENTRYPOINT [ "python", "/app/main.py" ]
