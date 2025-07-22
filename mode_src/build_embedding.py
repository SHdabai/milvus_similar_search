# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
# Time:  10:18
# FileName: build_embedding

# 在文件开头添加
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
import threading
import os
import logging
from typing import List, Tuple,Union
from FlagEmbedding import FlagModel, LightWeightFlagLLMReranker,FlagReranker
import torch
import numpy as np
import asyncio
from functools import partial
import multiprocessing

current_file_path = os.path.abspath(__file__)# 获取当前文件的绝对路径
current_dir = os.path.dirname(current_file_path)# 获取当前文件所在的目录

log_path = os.path.join(os.path.dirname(current_file_path), "../log", "run_log.log") #路径拼接
# 创建日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建文件处理器，将日志写入文件
# file_handler = logging.FileHandler('../log/stream_service.log')
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)

# 创建控制台处理器，将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)


model_path = os.path.join(current_dir, "../../BAAI", "bge-large-zh-v1.5") #路径拼接
# mode_path = os.path.join(current_dir, "../BAAI", "bge-m3") #路径拼接

logging.info(f'模型路徑....：{model_path}')

# 全局模型加载锁
model_lock = threading.Lock()

class VectorRetriever:
    '''保持原有类结构，修改初始化逻辑'''
    _shared_model = None

    def __init__(self, model_name: str = model_path):
        """延迟加载模型（线程安全）"""
        self.model_name = model_name
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        if VectorRetriever._shared_model is None:
            with model_lock:
                if VectorRetriever._shared_model is None:
                    pid = os.getpid()
                    logging.info(f"🛠️ [PID {pid}] 正在加载向量模型 : {self.model_name}...")
                    VectorRetriever._shared_model = FlagModel(
                        self.model_name,
                        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文档：",
                        use_fp16=True,
                        devices=['cuda:1'],
                    )
                    logging.info(f"✅ [PID {pid}] 向量模型加载完成.............")

        # 💡 确保无论如何 self.model 都指向共享模型
        if VectorRetriever._shared_model is None:
            raise RuntimeError("模型加载失败，_shared_model 仍然是 None！")

        self.model = VectorRetriever._shared_model

    async def query_embedding(
        self,
        query: List[str],
        batch_size: int = 32,
        max_length: int = 256
    ) -> Union[np.ndarray]:
        """异步执行向量生成"""
        try:
            if query is None:
                raise RuntimeError("请先传入 问题内容 构建问题向量.....！")

            loop = asyncio.get_running_loop()
            encode_func = partial(
                self.model.encode,
                sentences=query,
                batch_size=batch_size,
                max_length=max_length
            )
            query_embedding = await loop.run_in_executor(None, encode_func)
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)

            return query_embedding
        except Exception as e:
            return np.array([])



# 调试入口（你平时用于测试的方式）
async def main():
    qa_pairs = ['今天的阳光灿灿烂烂', '今天的阳光灿灿烂烂第三方收到']
    logging.info(f"🚀 初始化 {model_path} 模式检索系统...")

    retriever = VectorRetriever()
    a = await retriever.query_embedding(qa_pairs)

    logging.info("✨ 系统准备就绪........")
    print(f'问题输出向量：{a}')
    print(f'问题输出向量的维度：{len(a)}')
    print(f'问题输出向量的形状：{a.shape}')
    print(f'单个向量维度：{a.shape[1]}')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())