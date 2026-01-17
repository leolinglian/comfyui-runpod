FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN pip install --upgrade pip setuptools wheel

WORKDIR /comfyui
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && git checkout master

RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod>=1.5.0 xformers==0.0.23

COPY handler_simple.py /comfyui/handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

RUN mkdir -p /comfyui/output /comfyui/input

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

EXPOSE 8188

CMD ["/start.sh"]
