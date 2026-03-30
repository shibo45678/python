FROM ubuntu:latest
LABEL authors="shibo"


FROM python:3.11-slim

# 2. 设置容器内的工作目录 (保持工作目录在根， NeuralNetwork build)
WORKDIR /app

# 3. 先把 requirements.txt 复制进/app，并安装依赖
COPY deployment_package/single_lstm1_requirements.txt .
RUN pip install --no-cache-dir -r single_lstm1_requirements.txt

# 4. 把整个项目文件夹里的..都复制到容器的 /app 目录下（部署+src)
COPY deployment_package ./deployment_package
COPY src ./src
COPY deployment_test.py .
COPY data  ./data

#  将 /app 添加到 Python 模块搜索路径
ENV PYTHONPATH=/app

# 5. 入口脚本（可执行程序）
ENTRYPOINT ["python","/app/deployment_test.py"]

# 6. 指定容器启动时运行的命令（参数传递 脚本绝对路径）
#CMD ["--help"]

