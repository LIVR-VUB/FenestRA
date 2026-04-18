FROM nvcr.io/nvidia/pytorch:23.01-py3

LABEL Author="LIVR-VUB"
LABEL Version="1.1.0"
LABEL Description="DL Upsampling Benchmark (Stable Python 3.8 Environment - Docker Port)"

# Update system
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# 1. Install core dependencies with strict version pinning for stability
# numpy<1.24 to avoid 'float' attribute errors in older libs
RUN pip install --no-cache-dir \
    "numpy<1.24.0" \
    "scipy<1.11.0" \
    "scikit-image<0.20.0" \
    "pandas<2.0.0" \
    "PyYAML>=6.0" \
    "tifffile>=2023.0.0" \
    "matplotlib<3.8.0" \
    "seaborn<0.13.0" \
    "tqdm>=4.65.0"

# 2. Install opencv-headless specifically to avoid conflicts
# V4.8 is very stable with basicsr
RUN pip install --no-cache-dir "opencv-python-headless==4.8.0.74"

# 3. Install DL dependencies
# basicsr/realesrgan rely on older torch conventions sometimes
RUN pip install --no-cache-dir \
    "basicsr>=1.4.2" \
    "realesrgan>=0.3.0" \
    "cellpose>=2.2"

# 4. Install Image Quality Metrics
RUN pip install --no-cache-dir \
    pyiqa \
    lpips \
    brisque

# 5. Install SwinIR and HAT (from source)
WORKDIR /opt
RUN git clone https://github.com/JingyunLiang/SwinIR.git && \
    git clone https://github.com/XPixelGroup/HAT.git

# 6. FINAL CLEANUP: Force reinstall critical packages to ensure no unwanted upgrades happened
RUN pip install --force-reinstall --no-deps "numpy<1.24.0" "opencv-python-headless==4.8.0.74" || true
RUN pip uninstall -y opencv-python 2>/dev/null || true

# Add to Python path
ENV PYTHONPATH="/opt/SwinIR:/opt/HAT:${PYTHONPATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_VISIBLE_DEVICES=0

# Provide identical baseline run logic
ENTRYPOINT ["python"]
