ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"


RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC F60F4B3D7FA2AF80 && \
    apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone git@github.com:tad34s/Sperm-Parsing.git /sperm 
RUN mv /sperm/README.md /sperm/mmdetection-github/README.md
WORKDIR /sperm/mmdetection-github
# move epoch_35.pth to workdir
COPY epoch_35.pth .

ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir -e .

RUN python tools/test.py configs/git_fusionrcnn/cascade_mask_rcnn_r101_caffe_feature_blend_coarse_fine_edge_fpn_1x_spermparsingeval.py epoch_35.pth --eval bbox segm
