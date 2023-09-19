FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt update && apt install -y tmux vim htop

RUN pip install pytorch-lightning
RUN pip install torchtext torchvision ipython dash-core-components

WORKDIR /develop/results
WORKDIR /develop/data
WORKDIR /develop/code

