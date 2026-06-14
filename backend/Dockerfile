FROM python:3.12-slim

WORKDIR /app

# System libs required by OpenCV and numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch must be installed from PyTorch's own index before the rest
RUN pip install --no-cache-dir \
    torch==2.12.0+cpu \
    torchvision==0.27.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]