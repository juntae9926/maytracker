FROM nvcr.io/nvidia/tensorrt:23.10-py3

ARG python_version=3.11.4
ARG github_token

ENV TZ=Asia/Seoul

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential libffi-dev zlib1g-dev libssl-dev \
    libreadline-dev libncursesw5-dev libsqlite3-dev libgdbm-compat-dev \
    tk-dev libgdbm-dev libc6-dev libbz2-dev liblzma-dev pkg-config \
    git ssh vim libxrender1 ffmpeg libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python 3.11.4
RUN wget https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tgz && \
    tar xzf Python-${python_version}.tgz && \
    cd Python-${python_version} && \
    ./configure --prefix=/usr/local --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && rm -rf Python-${python_version}*

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Set default Python and pip versions
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.11 1

# Configure Git for cloning with token
RUN git config --global url."https://${github_token}@github.com/".insteadOf "https://github.com/"
RUN git config --global http.sslVerify false
