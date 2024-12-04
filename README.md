# maytracker

사내 Tracker 연구를 위한 라이브러리입니다.

## Dependencies

- Python ≥ 3.11
- CUDA ≥ 12.1
- PyTorch ≥ 2.1

## Dockerfile

To simplify installation, use the Dockerfile.

## Install Environment

1. Clone the repository:

    ```bash
    git clone https://github.com/mAy-I/maytracker.git --recurse-submodules
    ```

2. Run the Docker container:

    ```bash
    # Update github_token
    ARG github_token=""

    # Build and run the container
    docker build . -t maytracker
    docker run -v ~/videoset:/videoset -v ~/jt/maytracker:/user --ipc=host --gpus=all --name maytracker -it maytracker
    ```

3. Install Daram and Coram:

    ```bash
    # Install daram v3.18.3
    cd /user
    git clone -b v3.18.3 https://github.com/mAy-I/daram.git --recurse-submodules
    cd daram
    pip install -e .

    # Locate TensorRT packages in /usr/local/cuda
    cp /usr/lib/x86_64-linux-gnu/libnvinfer* /usr/local/cuda/targets/x86_64-linux/lib

    # Install coram
    cd /
    git clone https://github.com/mAy-I/coram.git
    export TENSORRT_DIR=/usr/local/cuda
    CORAM_WITH_OPS=1 CORAM_WITH_TRT=1 pip install -e ./coram
    ```

4. Install TrackEval
    ```bash
    cd TrackEval
    python setup.py develop
    ```

5. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Create TensorRT Engine

### Detector

Use the script in the `weights/` folder to create the TensorRT engine for the detector.

### Tracker

Use the script in the `weights/` folder to create the TensorRT engine for the tracker.

## Run Inference

1. Run End-to-End Evaluation:

    ```bash
    python run.py --eval
    ```

2. (Optional) Evaluate on all benchmarks
    ```bash
    python run.py
    python evaluate_all.py
    ```