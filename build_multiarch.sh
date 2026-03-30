#!/bin/bash

IMAGE_NAME="myapp"

# 初始化 buildx
docker buildx create --name multiarch --use 2>/dev/null || true
docker buildx inspect --bootstrap

# 构建并导出
docker buildx build --platform linux/arm64,linux/amd64 -t ${IMAGE_NAME}:latest --load .
docker save ${IMAGE_NAME}:latest | gzip > ${IMAGE_NAME}-multiarch.tar.gz

echo "✅ ${IMAGE_NAME}-multiarch.tar.gz ready (ARM64 + AMD64)"


# 使用
# 1. 给脚本执行权限
# chmod +x build-multiarch.sh

# 2. 运行构建
# ./build-multiarch.sh


# 在目标机器上使用
# 在任何机器上（ARM64 或 AMD64）加载镜像
# docker load < myapp-multiarch.tar.gz

# Docker 会自动选择适合当前机器的架构运行
# docker run myapp:latest