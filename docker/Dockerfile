FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
# python 3.10

ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

# python 3.10
ARG K2_VERSION="1.24.4.dev20240223+cuda11.8.torch2.1.0"
ARG KALDIFEAT_VERSION="1.25.4.dev20240223+cuda11.8.torch2.1.0"
ARG TORCHAUDIO_VERSION="2.1.0+cu118"

LABEL authors="Fangjun Kuang <csukuangfj@gmail.com>"
LABEL k2_version=${K2_VERSION}
LABEL kaldifeat_version=${KALDIFEAT_VERSION}
LABEL github_repo="https://github.com/k2-fsa/icefall"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        vim \
        tmux \
        htop \
    	libssl-dev \
        autoconf \
        automake \
        bzip2 \
        ca-certificates \
        ffmpeg \
        g++ \
        gfortran \
        git \
        libtool \
        make \
        patch \
        sox \
        subversion \
        unzip \
        valgrind \
        wget \
        zsh \
        zlib1g-dev \
        && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir \
      torchaudio==${TORCHAUDIO_VERSION} -f https://download.pytorch.org/whl/torchaudio/ \
      k2==${K2_VERSION} -f https://k2-fsa.github.io/k2/cuda.html \
      git+https://github.com/lhotse-speech/lhotse \
      kaldifeat==${KALDIFEAT_VERSION} -f https://csukuangfj.github.io/kaldifeat/cuda.html \
      kaldi_native_io \
      kaldialign \
      kaldifst \
      kaldilm \
      sentencepiece>=0.1.96 \
      tensorboard \
      typeguard \
      dill \
      onnx \
      onnxruntime \
      onnxmltools \
      onnxoptimizer \
      onnxsim \
      multi_quantization \
      typeguard \
      numpy \
      pytest \
      wandb \
      transformers \
      graphviz


RUN git clone https://github.com/k2-fsa/icefall /workspace/icefall && \
    cd /workspace/icefall && \
    pip install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir -U funasr
RUN pip3 install --no-cache-dir -U modelscope
RUN python -c 'from funasr import AutoModel; model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")'

ENV PYTHONPATH /workspace/icefall:$PYTHONPATH

ARG PW=1
RUN useradd -m asr --uid=1011 && echo "asr:1" | chpasswd
USER 1028:999

# Chuyển sang user mới này
USER $USER

WORKDIR /workspace/icefall

