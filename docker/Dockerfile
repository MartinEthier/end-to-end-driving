# Update image_version.txt when changes are made to Dockerfile 

# Original Dockerfile copied from:
# https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.4.0-cuda10.1-ubuntu16.04/Dockerfile
FROM nvidia/cuda:11.0-devel-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/root/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8 \
 && conda clean -ya

# Install PyTorch
RUN conda install -y pytorch torchvision cudatoolkit=11.0 -c pytorch \
 && conda clean -ya
 
# Install remaining libraries
RUN conda install -y -c conda-forge wandb matplotlib opencv \
 && conda clean -ya

WORKDIR /home
