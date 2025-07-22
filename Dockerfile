FROM crpi-1yvoh5oqfj561h3g.cn-hangzhou.personal.cr.aliyuncs.com/base-python/python:3.10

# 设置工作目录
WORKDIR /home/similar_search

# 拷贝依赖文件并安装
COPY requirements.txt ./
COPY FlagEmbedding ./FlagEmbedding

# RUN pip install -r requirements.txt -i http://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn
RUN pip install hypercorn
RUN pip install -r requirements.txt

# 拷贝代码（放在安装依赖之后，最大化利用缓存）
COPY . ./

# 可选：设置默认启动命令
# CMD ["python", "main.py"]
# 构建命令：docker build -t similar_search:dev-1.0 -f Dockerfile .
