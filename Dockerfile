FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

LABEL OS="Linux" \
      AUTHOR="XIANGRUI LIU" \
      EMAIL="xrliu.mail@gmail.com" \
      DESCRIPTION="Docker image for BCM-Net"

WORKDIR /home

RUN apt update && \
    # install python3.10 and pip3
    apt install software-properties-common python3 python3-pip -y && \
    # install pytorch 1.13.1 \
    pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    # install necessary python libraries
    pip3 install nibabel==5.1.0 tqdm==4.65.0 imageio==2.28.1 h5py==3.8.0 tensorboard==2.13.0 einops==0.6.1 timm==0.9.2 thop==0.1.1.post2209072238 torchac==0.9.3 compressai==1.2.4 opencv-python==4.7.0.72 && \
    # install ninja for torchac \
    apt install ninja-build -y && \
    # install compile tools
    apt install cmake git nano -y && \
    # download VVC reference software (VTM) and checkout to VTM-15.0
    cd /home && git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git && cd /home/VVCSoftware_VTM && git checkout VTM-15.0


