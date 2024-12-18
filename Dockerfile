FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    libspdlog-dev \
    libfmt-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY CMakeLists.txt ./
COPY src/ ./src/
COPY include/ ./include/
COPY python/ ./python/
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers \
    opencv-python \
    numpy \
    requests

# Build the C++ library
RUN mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && cd ..

# Install Python package
RUN pip install -e .

# Download models by default
RUN python scripts/download_models.py

# Set entry point to example
CMD ["python", "examples/simple_demo.py"] 