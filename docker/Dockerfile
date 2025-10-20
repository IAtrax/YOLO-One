FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git vim wget unzip build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone YOLO-One
RUN git clone https://github.com/IAtrax/YOLO-One.git /workspace/YOLO-One

WORKDIR /workspace/YOLO-One

# Install dependencies and package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Add to Python path
ENV PYTHONPATH="${PYTHONPATH}:/workspace/YOLO-One"

# Create startup script for COCO download
RUN echo '#!/bin/bash\n\
if [ ! -d "/workspace/YOLO-One/data/coco/train2017" ]; then\n\
    echo "📥 Downloading COCO dataset..."\n\
    mkdir -p /workspace/YOLO-One/data/coco\n\
    cd /workspace/YOLO-One/data/coco\n\
    wget -q http://images.cocodataset.org/zips/train2017.zip &\n\
    wget -q http://images.cocodataset.org/zips/val2017.zip &\n\
    wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip &\n\
    wait\n\
    echo "📦 Extracting..."\n\
    unzip -q train2017.zip &\n\
    unzip -q val2017.zip &\n\
    unzip -q annotations_trainval2017.zip &\n\
    wait\n\
    rm *.zip\n\
    echo "✅ COCO ready!"\n\
else\n\
    echo "✅ COCO already exists!"\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Verify installation
RUN python -c "import yolo_one; print('✅ YOLO-One ready')" && \
    python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"

WORKDIR /workspace/YOLO-One

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]