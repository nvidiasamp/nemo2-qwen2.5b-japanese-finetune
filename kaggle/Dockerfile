# 基于NVIDIA CUDA 12.2和Ubuntu 20.04的基础镜像
FROM nvidia/cuda:12.2-cudnn8-devel-ubuntu20.04

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libsndfile1 \
    libsndfile1-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    vim \
    wget \
    unzip \
    htop \
    tmux \
    screen \
    apt-transport-https \
    software-properties-common \
    bzip2 \
    less \
    libgl1 \
    openssh-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装Intel MKL优化库(提升CPU性能)
RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-mkl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 配置环境变量
ENV PATH="/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
# 配置Python不生成__pycache__文件，节省空间
ENV PYTHONDONTWRITEBYTECODE=1
# 设置时区
ENV TZ=Asia/Shanghai

# 安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -ya

# 创建Python 3.10.13环境
RUN conda create -n kaggle python=3.10.13 -y \
    && conda clean -ya

# 激活Kaggle环境并安装基础依赖
SHELL ["/bin/bash", "-c"]
RUN echo "source activate kaggle" > ~/.bashrc
ENV PATH /opt/conda/envs/kaggle/bin:$PATH

# 安装深度学习核心库
RUN pip install --no-cache-dir \
    torch==2.2.2+cu121 \
    torchvision==0.17.2+cu121 \
    torchaudio==2.2.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 安装TensorFlow
RUN pip install --no-cache-dir \
    tensorflow==2.16.1 \
    tensorflow-io \
    tensorboard

# 安装ONNX Runtime
RUN pip install --no-cache-dir \
    onnx \
    onnxruntime-gpu

# 安装机器学习工具
RUN pip install --no-cache-dir \
    scikit-learn \
    xgboost \
    lightgbm \
    catboost \
    pandas==2.2.2 \
    numpy==1.26.4 \
    polars \
    matplotlib \
    seaborn \
    plotly \
    networkx \
    statsmodels \
    scipy

# 安装大语言模型与RAG相关库
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    langchain \
    faiss-cpu \
    chromadb \
    sentence-transformers \
    datasets

# 安装图像处理库
RUN pip install --no-cache-dir \
    opencv-python \
    pillow \
    albumentations \
    scikit-image \
    kornia \
    timm

# 安装音频处理库
RUN pip install --no-cache-dir \
    librosa \
    audiomentations \
    pydub \
    wavfile

# 安装实验跟踪与管理工具
RUN pip install --no-cache-dir \
    wandb \
    dvc \
    dvc-gdrive \
    mlflow \
    tensorboardx \
    optuna

# 安装开发环境与辅助工具
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    papermill \
    tqdm \
    colorama \
    black \
    isort \
    pytest \
    ipython \
    nvidia-dali-cuda120

# 创建监控和健康检查脚本
COPY ./scripts/monitor_gpu.py /workspace/monitor_gpu.py

# 配置Jupyter Lab
RUN jupyter labextension install jupyterlab-plotly \
    && mkdir -p /root/.jupyter/ \
    && echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py

# 创建工作目录结构
RUN mkdir -p /workspace/data \
    && mkdir -p /workspace/models \
    && mkdir -p /workspace/notebooks \
    && mkdir -p /workspace/src \
    && mkdir -p /workspace/outputs

# 设置启动脚本
COPY ./scripts/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# 创建一个示例GPU内存监控脚本
RUN echo 'import os\nimport time\nimport subprocess\nimport numpy as np\nimport psutil\n\ndef get_gpu_memory_info():\n    try:\n        result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"])\n        memory_info = [int(x) for x in result.decode("utf-8").split(",")]\n        return memory_info\n    except Exception as e:\n        print(f"Error getting GPU info: {str(e)}")\n        return [0, 0]\n\ndef monitor_resources():\n    while True:\n        # CPU使用率\n        cpu_percent = psutil.cpu_percent(interval=1)\n        # 内存使用率\n        memory = psutil.virtual_memory()\n        memory_percent = memory.percent\n        # GPU内存\n        gpu_memory = get_gpu_memory_info()\n        gpu_used = gpu_memory[0]\n        gpu_total = gpu_memory[1]\n        gpu_percent = (gpu_used / gpu_total) * 100 if gpu_total > 0 else 0\n        \n        print(f"CPU使用率: {cpu_percent}%")\n        print(f"内存使用率: {memory_percent}% ({memory.used / (1024**3):.2f}GB / {memory.total / (1024**3):.2f}GB)")\n        print(f"GPU内存使用率: {gpu_percent:.2f}% ({gpu_used / 1024:.2f}GB / {gpu_total / 1024:.2f}GB)")\n        print("-" * 50)\n        \n        # 如果GPU内存使用率超过90%，发出警告\n        if gpu_percent > 90:\n            print("警告: GPU内存使用率过高!")\n        \n        time.sleep(5)\n\nif __name__ == "__main__":\n    print("开始监控系统资源...")\n    monitor_resources()' > /workspace/monitor_gpu.py

# 配置环境变量优化性能
ENV OMP_NUM_THREADS=12
ENV MKL_NUM_THREADS=12
ENV NUMEXPR_NUM_THREADS=12
ENV OPENBLAS_NUM_THREADS=12
ENV VECLIB_MAXIMUM_THREADS=12
ENV TOKENIZERS_PARALLELISM=true
ENV PYTHONUNBUFFERED=1
# 默认只使用A6000 GPU（假定A6000在索引位置0）
ENV CUDA_VISIBLE_DEVICES=0

# 设置容器启动命令
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
