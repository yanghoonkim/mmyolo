FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y git vim tmux htop python3-pip software-properties-common ffmpeg libsm6 libxext6 wget

RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt update && apt install -y python3.8

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install openmim future tensorboard pandas notebook==6.4.8 traitlets==5.9.0 polars

RUN mim install "mmengine>=0.6.0"

RUN mim install "mmcv>=2.0.0rc4,<2.1.0"

RUN mim install "mmdet>=3.0.0,<4.0.0"

WORKDIR /root

RUN git clone https://github.com/yanghoonkim/mmyolo.git

WORKDIR /root/mmyolo

RUN git checkout nia

RUN pip install -r requirements/albu.txt

RUN mim install -v -e .

RUN wget https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth -P nia/

CMD bash
