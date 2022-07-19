# 單機教學
GitLab : [https://gitlab.com/cmuh-aic/harmonia.git](https://gitlab.com/cmuh-aic/harmonia.git)

## Clone
先取得原始harmonia
```
git clone https://gitlab.com/cmuh-aic/harmonia.git
```

## Build
建立`harmonia/operator`的 docker image
```
make all
```

## Example
首先進入 example 資料夾
### Generate gRPC Python Modules
```
make -C ../src/protos python_protos
```
### Build Application Images
建立 edge 的 docker image
```
cp -pv ../src/protos/python_protos/* edge
docker build -t mnist_edge edge
```

### Docker Deployment
接著進入 docker_deployment 資料夾
1. 建立 `mnist` docker network
```
docker network create mnist
```
2. Create a Gitea instance.
```bash
docker run -d \
    --env LFS_START_SERVER=true \
    --env INSTALL_LOCK=true \
    --env ROOT_URL=http://gitea:3000 \
    --publish 3000:3000 \
    --network mnist \
    --name gitea \
    gitea/gitea:1.15.1

# Notes:
# `LFS_START_SERVER` enables git lfs in Gitea which is required by Harmonia.
# `ROOT_URL` shuould be set the same as container name to ensure git operation work.
```

3. Setup Gitea for the MNIST example:  
```bash
docker cp ./gitea_setup.sh gitea:/gitea_setup.sh
docker exec gitea bash /gitea_setup.sh
```

Including creates
* admin account: `gitea` (password: `password`)
* user accounts: `aggregator` `edge1` `edge2` `logserver`
* repositories: `train-plan` `global-model` `local-model1` `local-model2`
* repository permissions: TODO
* webhooks:
    * `train-plan` to `http://aggregator:9080` `http://edge1:9080` `http://edge2:9080`
    * `global-model` to `http://edge1:9080` `http://edge2:9080` `http://logserver:9080`
    * `local-model1` to `http://aggregator:9080` `http://logserver:9080`
    * `local-model2` to `http://aggregator:9080` `http://logserver:9080`

4. Push the pretrained model to `global-model`.
```bash
docker network connect bridge gitea

git clone http://gitea@localhost:3000/gitea/global-model.git
pushd global-model

git commit -m "pretrained model" --allow-empty
git push origin master

popd
rm -rf global-model
```

5. Deploy every FL participants by `docker-compose up` in each folder
```bash
pushd aggregator
docker-compose up -d
popd

pushd edge1
docker-compose up -d
popd

pushd edge2
docker-compose up -d
popd

pushd logserver
docker-compose up -d
popd
```

6. Push a train plan to trigger federated MNIST.
```bash
# bash
# docker network connect bridge gitea
# 因為前面有connect過了

git clone http://gitea@localhost:3000/gitea/train-plan.git
pushd train-plan

cat > plan.json << EOF
{
    "name": "MNIST",
    "round": 10,
    "edge": 2,
    "EpR": 1,
    "timeout": 86400,
    "pretrainedModel": "master"
}
EOF

git add plan.json
git commit -m "train plan commit"
git push origin master

popd
rm -rf train-plan
```
You can check FL result within Gitea web UI (http://localhost:3000) or Tensorboard UI (http://localhost:6006).

# 多機教學(兩個VM)
以下使用10.24.211.111、161兩台伺服器示範

基本上跟單機大同小異，但需要做以下設定:

### 更改`gitea_setup.sh`
將 `webhook` 的 `URL` 做以下更改
- edge1:9080 => 10.24.211.161:9082
- edge2:9080 => 10.24.211.111:9080
- logserver:9080 => 10.24.211.161:9081
- aggregator:9080 => 10.24.211.161:9080
```bash
...

# Create webhooks

# train-plan
curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9080"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/train-plan/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9082"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/train-plan/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.111:9080"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/train-plan/hooks

# global-model
curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9082"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/global-model/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.111:9080"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/global-model/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9081"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/global-model/hooks

# local-model1
curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9080"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/local-model1/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9081"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/local-model1/hooks

# local-model2
curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9080"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/local-model2/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9081"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/local-model2/hooks
```

### 更改各個 `docker-compose.yml` 及 `config.yml` 

- aggregator
	`docker-compose.yml`
    ```yml
    version: "3.7"
    services:
        app:
        image: harmonia/fedavg
        environment:
            OPERATOR_URI: operator:8787
        volumes:
            - type: volume
            source: shared
            target: /repos
        operator:
        image: harmonia/operator
        ports:
            - '9080:9080'
        volumes:
            - ./config.yml:/app/config.yml
            - type: volume
            source: shared
            target: /repos
    volumes:
        shared:
    ```

    `config.yml`
    ```yml
    type: aggregator
    logLevel: debug
    notification:
        type: push
        stewardServerURI: ":9080"
    operatorGrpcServerURI: operator:8787
    appGrpcServerURI: app:7878
    gitUserToken: 1qaz_WSX
    aggregatorModelRepo:
        gitHttpURL: http://aggregator@10.24.211.161:3000/gitea/global-model.git
    edgeModelRepos:
      - gitHttpURL: http://aggregator@10.24.211.161:3000/gitea/local-model1.git
      - gitHttpURL: http://aggregator@10.24.211.161:3000/gitea/local-model2.git
    trainPlanRepo:
        gitHttpURL: http://aggregator@10.24.211.161:3000/gitea/train-plan.git
    ```

- edge1	
	`docker-compose.yml`
    ```yml
    version: "3.7"
    services:
      app:
        image: mnist_edge
        runtime: nvidia
        environment:
          OPERATOR_URI: operator:8787
          NVIDIA_DRIVER_CAPABILITIES: compute,utility
          NVIDIA_VISIBLE_DEVICES: all
        volumes:
          - type: volume
            source: shared
            target: /repos
      operator:
        image: harmonia/operator
        ports:
          - '9082:9080'
        volumes:
          - ./config.yml:/app/config.yml
          - type: volume
            source: shared
            target: /repos
    volumes:
      shared:
    ```

    `config.yml`
    ```yml
    type: edge
    logLevel: debug
    notification:
      type: push
      stewardServerURI: ":9080"
    operatorGrpcServerURI: operator:8787
    appGrpcServerURI: app:7878
    gitUserToken: 1qaz_WSX
    aggregatorModelRepo:
      gitHttpURL: http://edge1@10.24.211.161:3000/gitea/global-model.git
    edgeModelRepo:
      gitHttpURL: http://edge1@10.24.211.161:3000/gitea/local-model1.git
    trainPlanRepo:
      gitHttpURL: http://edge1@10.24.211.161:3000/gitea/train-plan.git
    ```

- edge2
	`docker-compose.yml`
    ```yml
    version: "3.7"
    services:
      app:
        image: mnist_edge
        runtime: nvidia
        ports:
          - '7878:7878'
        environment:
          OPERATOR_URI: operator:8787
          NVIDIA_DRIVER_CAPABILITIES: compute,utility
          NVIDIA_VISIBLE_DEVICES: all
        volumes:
          - type: volume
            source: shared
            target: /repos
      operator:
        image: harmonia/operator
        ports:
          - '9080:9080'
          - '8787:8787'
        volumes:
          - ./config.yml:/app/config.yml
          - type: volume
            source: shared
            target: /repos
    volumes:
      shared:
    ```

    `config.yml`
    ```yml
    type: edge
    logLevel: debug
    notification:
      type: push
      stewardServerURI: ":9080"
    operatorGrpcServerURI: operator:8787
    appGrpcServerURI: app:7878
    gitUserToken: 1qaz_WSX
    aggregatorModelRepo:
      gitHttpURL: http://edge2@10.24.211.161:3000/gitea/global-model.git
    edgeModelRepo:
      gitHttpURL: http://edge2@10.24.211.161:3000/gitea/local-model2.git
    trainPlanRepo:
      gitHttpURL: http://edge2@10.24.211.161:3000/gitea/train-plan.git
    ```

- logserver
	`docker-compose.yml`
    ```yml
    version: "3.7"
    services:
      tensorboard:
        image: tensorflow/tensorflow
        entrypoint: ["tensorboard", "--logdir=/tensorboard_data", "--bind_all"]
        ports:
          - "6006:6006"
        volumes:
          - type: volume
            source: shared
            target: /tensorboard_data
      operator:
        image: harmonia/logserver
        ports:
          - '9081:9080'
        volumes:
          - ./config.yml:/app/config.yml
          - type: volume
            source: shared
            target: /tensorboard_data
    volumes:
      shared:
    ```

    `config.yml`
    ```yml
    stewardServerURI: "0.0.0.0:9080"
    gitUserToken: 1qaz_WSX
    tensorboardDataRootDir: /tensorboard_data
    modelRepos:
      - gitHttpURL: http://logserver@10.24.211.161:3000/gitea/global-model.git
      - gitHttpURL: http://logserver@10.24.211.161:3000/gitea/local-model1.git
      - gitHttpURL: http://logserver@10.24.211.161:3000/gitea/local-model2.git

    ```
### Docker Deployment
1. Create a Gitea instance.
```bash
docker run -d \
    --env LFS_START_SERVER=true \
    --env INSTALL_LOCK=true \
    --env ROOT_URL=http://192.168.1.28:3000 \
    --publish 3000:3000 \
    --name gitea \
    gitea/gitea:1.15.1

# Notes:
# `LFS_START_SERVER` enables git lfs in Gitea which is required by Harmonia.
# `ROOT_URL` shuould be set the same as container name to ensure git operation work.
```

2. Setup Gitea for the MNIST example:  
```bash
docker cp ./gitea_setup.sh gitea:/gitea_setup.sh
docker exec gitea bash /gitea_setup.sh
```

Including creates
* admin account: `gitea` (password: `password`)
* user accounts: `aggregator` `edge1` `edge2` `logserver`
* repositories: `train-plan` `global-model` `local-model1` `local-model2`
* repository permissions: TODO
* webhooks:
    * `train-plan` to `http://10.24.211.161:9080` `http://10.24.211.161:9082` `http://10.24.211.111:9080`
    * `global-model` to `http://10.24.211.161:9082` `http://10.24.211.111:9080` `http://10.24.211.161:9081`
    * `local-model1` to `http://10.24.211.161:9080` `http://10.24.211.161:9081`
    * `local-model2` to `http://10.24.211.161:9080` `http://10.24.211.161:9081`

3. Push the pretrained model to `global-model`.
```bash
docker network connect bridge gitea

git clone http://gitea@localhost:3000/gitea/global-model.git
pushd global-model

git commit -m "pretrained model" --allow-empty
git push origin master

popd
rm -rf global-model
```

4. Deploy every FL participants by `docker-compose up` in each folder
- 10.24.211.161
    ```bash
    pushd aggregator
    docker-compose up -d
    popd

    pushd edge1
    docker-compose up -d
    popd

    pushd logserver
    docker-compose up -d
    popd
    ```
- 10.24.211.111
    ```bash
    pushd edge2
    docker-compose up -d
    popd
    ```

5. Push a train plan to trigger federated MNIST.
```bash
# bash
# docker network connect bridge gitea
# 因為前面有connect過了

git clone http://gitea@localhost:3000/gitea/train-plan.git
pushd train-plan

cat > plan.json << EOF
{
    "name": "MNIST",
    "round": 10,
    "edge": 2,
    "EpR": 1,
    "timeout": 86400,
    "pretrainedModel": "master"
}
EOF

git add plan.json
git commit -m "train plan commit"
git push origin master

popd
rm -rf train-plan
```
You can check FL result within Gitea web UI (http://10.24.211.161:3000) or Tensorboard UI (http://10.24.211.161:6006).

# 多機教學(三個VM)
以下使用10.24.211.111、161兩台伺服器示範

基本上跟單機大同小異，但需要做以下設定:

### 更改`gitea_setup.sh`
將 `webhook` 的 `URL` 做以下更改
- edge1:9080 => 192.168.1.28:9080
- edge2:9080 => 192.168.1.29:9080
- logserver:9080 => 192.168.1.27:9081
- aggregator:9080 => 192.168.1.27:9080
```bash
...

# Create webhooks

# train-plan
curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9080"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/train-plan/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9082"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/train-plan/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.111:9080"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/train-plan/hooks

# global-model
curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9082"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/global-model/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.111:9080"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/global-model/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9081"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/global-model/hooks

# local-model1
curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9080"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/local-model1/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9081"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/local-model1/hooks

# local-model2
curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9080"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/local-model2/hooks

curl -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "config": {"content_type": "json", "url": "http://10.24.211.161:9081"}, "events": ["push"], "type": "gitea"}' \
    http://gitea:password@127.0.0.1:3000/api/v1/repos/gitea/local-model2/hooks
```

### 更改各個 `docker-compose.yml` 及 `config.yml` 

- aggregator
	`docker-compose.yml`
    ```yml
    version: "3.7"
    services:
        app:
        image: harmonia/fedavg
        environment:
            OPERATOR_URI: operator:8787
        volumes:
            - type: volume
            source: shared
            target: /repos
        operator:
        image: harmonia/operator
        ports:
            - '9080:9080'
        volumes:
            - ./config.yml:/app/config.yml
            - type: volume
            source: shared
            target: /repos
    volumes:
        shared:
    ```

    `config.yml`
    ```yml
    type: aggregator
    logLevel: debug
    notification:
        type: push
        stewardServerURI: ":9080"
    operatorGrpcServerURI: operator:8787
    appGrpcServerURI: app:7878
    gitUserToken: 1qaz_WSX
    aggregatorModelRepo:
        gitHttpURL: http://aggregator@10.24.211.161:3000/gitea/global-model.git
    edgeModelRepos:
      - gitHttpURL: http://aggregator@10.24.211.161:3000/gitea/local-model1.git
      - gitHttpURL: http://aggregator@10.24.211.161:3000/gitea/local-model2.git
    trainPlanRepo:
        gitHttpURL: http://aggregator@10.24.211.161:3000/gitea/train-plan.git
    ```

- edge1	
	`docker-compose.yml`
    ```yml
    version: "3.7"
    services:
      app:
        image: mnist_edge
        runtime: nvidia
        environment:
          OPERATOR_URI: operator:8787
          NVIDIA_DRIVER_CAPABILITIES: compute,utility
          NVIDIA_VISIBLE_DEVICES: all
        volumes:
          - type: volume
            source: shared
            target: /repos
      operator:
        image: harmonia/operator
        ports:
          - '9082:9080'
        volumes:
          - ./config.yml:/app/config.yml
          - type: volume
            source: shared
            target: /repos
    volumes:
      shared:
    ```

    `config.yml`
    ```yml
    type: edge
    logLevel: debug
    notification:
      type: push
      stewardServerURI: ":9080"
    operatorGrpcServerURI: operator:8787
    appGrpcServerURI: app:7878
    gitUserToken: 1qaz_WSX
    aggregatorModelRepo:
      gitHttpURL: http://edge1@10.24.211.161:3000/gitea/global-model.git
    edgeModelRepo:
      gitHttpURL: http://edge1@10.24.211.161:3000/gitea/local-model1.git
    trainPlanRepo:
      gitHttpURL: http://edge1@10.24.211.161:3000/gitea/train-plan.git
    ```

- edge2
	`docker-compose.yml`
    ```yml
    version: "3.7"
    services:
      app:
        image: mnist_edge
        runtime: nvidia
        ports:
          - '7878:7878'
        environment:
          OPERATOR_URI: operator:8787
          NVIDIA_DRIVER_CAPABILITIES: compute,utility
          NVIDIA_VISIBLE_DEVICES: all
        volumes:
          - type: volume
            source: shared
            target: /repos
      operator:
        image: harmonia/operator
        ports:
          - '9080:9080'
          - '8787:8787'
        volumes:
          - ./config.yml:/app/config.yml
          - type: volume
            source: shared
            target: /repos
    volumes:
      shared:
    ```

    `config.yml`
    ```yml
    type: edge
    logLevel: debug
    notification:
      type: push
      stewardServerURI: ":9080"
    operatorGrpcServerURI: operator:8787
    appGrpcServerURI: app:7878
    gitUserToken: 1qaz_WSX
    aggregatorModelRepo:
      gitHttpURL: http://edge2@10.24.211.161:3000/gitea/global-model.git
    edgeModelRepo:
      gitHttpURL: http://edge2@10.24.211.161:3000/gitea/local-model2.git
    trainPlanRepo:
      gitHttpURL: http://edge2@10.24.211.161:3000/gitea/train-plan.git
    ```

- logserver
	`docker-compose.yml`
    ```yml
    version: "3.7"
    services:
      tensorboard:
        image: tensorflow/tensorflow
        entrypoint: ["tensorboard", "--logdir=/tensorboard_data", "--bind_all"]
        ports:
          - "6006:6006"
        volumes:
          - type: volume
            source: shared
            target: /tensorboard_data
      operator:
        image: harmonia/logserver
        ports:
          - '9081:9080'
        volumes:
          - ./config.yml:/app/config.yml
          - type: volume
            source: shared
            target: /tensorboard_data
    volumes:
      shared:
    ```

    `config.yml`
    ```yml
    stewardServerURI: "0.0.0.0:9080"
    gitUserToken: 1qaz_WSX
    tensorboardDataRootDir: /tensorboard_data
    modelRepos:
      - gitHttpURL: http://logserver@10.24.211.161:3000/gitea/global-model.git
      - gitHttpURL: http://logserver@10.24.211.161:3000/gitea/local-model1.git
      - gitHttpURL: http://logserver@10.24.211.161:3000/gitea/local-model2.git

    ```
### Docker Deployment
1. Create a Gitea instance.
```bash
docker run -d \
    --env LFS_START_SERVER=true \
    --env INSTALL_LOCK=true \
    --env ROOT_URL=http://192.168.1.27:3000 \
    --publish 3000:3000 \
    --name gitea \
    gitea/gitea:1.15.1

# Notes:
# `LFS_START_SERVER` enables git lfs in Gitea which is required by Harmonia.
# `ROOT_URL` shuould be set the same as container name to ensure git operation work.
```

2. Setup Gitea for the MNIST example:  
```bash
docker cp ./gitea_setup.sh gitea:/gitea_setup.sh
docker exec gitea bash /gitea_setup.sh
```

Including creates
* admin account: `gitea` (password: `password`)
* user accounts: `aggregator` `edge1` `edge2` `logserver`
* repositories: `train-plan` `global-model` `local-model1` `local-model2`
* repository permissions: TODO
* webhooks:
    * `train-plan` to `http://10.24.211.161:9080` `http://10.24.211.161:9082` `http://10.24.211.111:9080`
    * `global-model` to `http://10.24.211.161:9082` `http://10.24.211.111:9080` `http://10.24.211.161:9081`
    * `local-model1` to `http://10.24.211.161:9080` `http://10.24.211.161:9081`
    * `local-model2` to `http://10.24.211.161:9080` `http://10.24.211.161:9081`

3. Push the pretrained model to `global-model`.
```bash
docker network connect bridge gitea

git clone http://gitea@localhost:3000/gitea/global-model.git
pushd global-model

git commit -m "pretrained model" --allow-empty
git push origin master

popd
rm -rf global-model
```

4. Deploy every FL participants by `docker-compose up` in each folder
- 192.168.1.27
    ```bash
    pushd aggregator
    docker-compose up -d
    popd
    
    pushd logserver
    docker-compose up -d
    popd
    ```
- 192.168.1.28   
    ```bash
    pushd edge1
    docker-compose up -d
    popd
    ```
- 192.168.1.29
    ```bash
    pushd edge2
    docker-compose up -d
    popd
    ```

5. Push a train plan to trigger federated MNIST.
```bash
# bash
# docker network connect bridge gitea
# 因為前面有connect過了

git clone http://gitea@localhost:3000/gitea/train-plan.git
pushd train-plan

cat > plan.json << EOF
{
    "name": "MNIST",
    "round": 10,
    "edge": 2,
    "EpR": 1,
    "timeout": 86400,
    "pretrainedModel": "master"
}
EOF

git add plan.json
git commit -m "train plan commit"
git push origin master

popd
rm -rf train-plan
```
You can check FL result within Gitea web UI (http://10.24.211.161:3000) or Tensorboard UI (http://10.24.211.161:6006).
