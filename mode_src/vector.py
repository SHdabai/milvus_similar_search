# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
# Time:  15:17
# FileName: main
# Tools: PyCharm
# 在文件开头添加
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
import gc  # 需要引入垃圾回收模块

import os
import time
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from scipy.spatial.distance import cdist
from FlagEmbedding import FlagModel, LightWeightFlagLLMReranker,FlagReranker
import os



current_file_path = os.path.abspath(__file__)# 获取当前文件的绝对路径
current_dir = os.path.dirname(current_file_path)# 获取当前文件所在的目录

log_path = os.path.join(os.path.dirname(current_file_path), "../log", "run_log.log") #路径拼接
# 创建日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 创建文件处理器，将日志写入文件
file_handler = logging.FileHandler('../log/stream_service.log')
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


# mode_path = os.path.join(current_dir, "BAAI", "bge-large-zh-v1.5") #路径拼接
mode_path = os.path.join(current_dir, "../BAAI", "bge-m3") #路径拼接
rerank_path = os.path.join(current_dir, "../BAAI", "bge-reranker-v2-m3") #路径拼接

logging.info(f'模型路徑....：{mode_path}')

class VectorRetriever:
    '''
    VectorRetriever (向量检索器)
        使用 FlagModel 生成问题向量
        构建向量索引快速检索
        适合大规模快速初筛

    '''

    def __init__(self, model_name: str = mode_path):
        """
        初始化向量检索器
        :param model_name: 向量模型名称
        """
        logging.info(f"🛠️ 正在加载向量模型 : {model_name}...")
        self.model = FlagModel(
            model_name,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文档：",
            use_fp16=True, # Setting use_fp16 to True speeds up computation with a slight performance degradation
            devices = ['cuda:1']
        )
        self.index_embeddings = None
        self.qa_pairs = []
        logging.info("✅ 向量模型加载完成.............")

    def build_index(self, qa_pairs: List[Tuple[str, str]]):
        """
        构建问答对向量索引
        :param qa_pairs: 问答对列表，格式为[(问题1, 答案1), (问题2, 答案2), ...]
        """
        if not qa_pairs:
            raise ValueError("知识库不能为空！")


        logging.info(f"🔧 正在构建 {len(qa_pairs)} 条问答对的索引...")
        self.qa_pairs = qa_pairs  #問答對內容

        # 提取所有问题文本
        questions = [q for q, _ in qa_pairs]

        print(f'構建的所有問題內容........：{questions}')

        # 批量编码所有问题 questions ====> [str,str,...]
        self.index_embeddings = self.model.encode(questions,
                                                  batch_size=128,
                                                  max_length=64,
                                                  )

        logging.info(f"✅ 索引构建完成.............形状：{self.index_embeddings.shape}")
        print(f"✅ 索引內容：{self.index_embeddings}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Tuple[str, str]]]:
        """
        执行向量相似度搜索
        :param query: 用户查询
        :param top_k: 返回的结果数量
        :return: 包含(分数, (问题, 答案))的列表
        """
        if self.index_embeddings is None:
            raise RuntimeError("请先调用 build_index 构建索引！")

        # 编码查询
        query_embedding = self.model.encode(
            [query],
        )
        print(f"✅ 問題向量：{query_embedding.shape}") # ====>数据格式 [[]]
        print(f"✅ 問題向量转置T后的形状：{query_embedding.T.shape}") # ====>数据格式 [[]]
        # print(f"✅ 問題向量：{query_embedding[0]}") #====>向量数据格式 []


        # 三级精度优化:
        n = len(self.index_embeddings)
        # 1. 小数据集快速方案
        if n < 5000:
            # 向量化点积 (最优方案)
            scores = self.index_embeddings @ query_embedding.T
            '''
            elf.index_embeddings：形状为 (n, d)，其中：
                                n 是索引向量的数量（文档数量）
                                d 是向量的维度
            query_embedding：形状为 (1, d)（单个查询）或 (m, d)（多个查询）
                                m 是查询向量的数量
                                d 是相同的向量维度
            '''
        # 2. 中等规模优化
        elif n < 100000:
            # BLAS优化点积 (避免临时变量)
            scores = np.dot(self.index_embeddings, query_embedding.T)
        # 3. 超大规模方案
        else:
            # 分块计算余弦距离
            scores = -cdist(self.index_embeddings,
                            [query_embedding],
                            'cosine').flatten()

        # 获取TopK结果
        # top_indices = np.argsort(scores)[::-1][:top_k] # 不使用矩阵计算，不进行矩阵转置 T
        top_indices = np.argsort(scores.flatten())[::-1][:top_k]
        '''
        np.argsort():
            返回的是排序后的索引，从小到大排列
            分数最低的在前面：索引 2（0.75）→ 索引 0（0.8）→ 索引 1（0.95）
        scores.flatten():
            展平数组:我们可以使用scores.flatten()或np.squeeze()将其转换为一维数组
        [::-1]:
            反转数组，变成从大到小排列
            现在最高分在前面：索引 1（0.95）→ 索引 0（0.8）→ 索引 2（0.75）
        [:top_k]:
            取前 k 个元素
        '''
        # print(f'计算得到的分数====>:{scores}')
        # print(f'np.argsort(scores.flatten())====>:{np.argsort(scores.flatten())}')
        # print(f'scores.flatten()====>:{scores.flatten()}')
        # print(f'获取top个结果====>:{top_indices}')
        results = [(float(scores[i]), self.qa_pairs[i]) for i in top_indices]
        return results


class QASystem:
    '''
    QASystem (问答系统)
    统一接口支持三种模式
    提供完整检索流程
    输出带格式的结果

    '''

    def __init__(self, qa_pairs: List[Tuple[str, str]], mode: str = "hybrid"):
        """
        初始化问答系统
        :param qa_pairs: 问答对列表
        :param mode: 检索模式 (vector, rerank, hybrid)
        """
        self.mode = mode
        self.qa_pairs = qa_pairs
        self.retriever = None

        # 根据模式选择检索器
        if mode == "vector":  #純向量
            self.retriever = VectorRetriever()
        else:
            raise ValueError(f"不支持的模式: {mode}")

        # 构建索引
        logging.info(f"🚀 初始化 {mode} 模式检索系统...")
        self.retriever.build_index(qa_pairs)
        logging.info("✨ 系统准备就绪........")

    def query(self, user_question: str, top_k: int = 10):
        """
        执行查询并返回结果
        :param user_question: 用户提问
        :param top_k: 返回结果数量
        """
        if not self.retriever:
            raise RuntimeError("检索系统未初始化！")

        print(f"\n🔎 查询: {user_question}")
        start_time = time.time()

        # 执行检索
        results = self.retriever.search(user_question, top_k=top_k)

        latency = (time.time() - start_time) * 1000
        print(f"⏱️ 检索耗时: {latency:.2f}ms")

        # 打印结果
        print("\n📊 搜索结果:")
        for rank, (score, (q, a)) in enumerate(results, 1):
            print(f"【第{rank}名】(相关性: {score:.4f})")
            print(f"  问题: {q}")
            print(f"  答案: {a[:120]}{'...' if len(a) > 120 else ''}")
            print("─" * 80)

        return results


import random
from typing import List, Tuple


def generate_sample_qa(num_pairs: int = 1000) -> List[Tuple[str, str]]:
    '''
    生成示例问答对用于测试

    Args:
        num_pairs: 生成的问答对数量

    Returns:
        问答对列表，每个元素是(问题, 答案)元组
    '''
    print(f"生成 {num_pairs} 条示例问答对...")

    # 基础问题模板
    templates = [
        ("如何{}{}", "{}的具体步骤：1.准备材料 2.完成{} 3.检查结果"),
        ("什么是{}", "{}是指{}. 它常用于{}场景"),
        ("{}怎么安装", "安装方法：1.下载安装包 2.运行{} 3.按指引操作"),
        ("{}多少钱", "价格范围从{}{}{}元不等"),
        ("{}有什么功能", "核心功能包括：功能1{} 功能2{}"),
    ]

    # 示例关键词
    subjects = ["手机", "电脑", "软件", "系统", "支付", "账户", "会员", "服务", "产品", "物流"]
    verbs = ["设置", "使用", "购买", "开通", "激活", "取消", "配置", "下载", "连接"]
    descriptors = ["简单", "快速", "高效", "专业", "安全", "便捷", "稳定", "高级"]  # 新增描述词列表

    # 生成问题对
    qa_list = []

    for i in range(num_pairs):
        # 随机选择模板和关键词
        template_idx = random.randint(0, len(templates) - 1)
        subject = random.choice(subjects)
        verb = random.choice(verbs)

        # 获取问题和答案模板
        q_template, a_template = templates[template_idx]

        # 根据模板生成问题
        question = q_template.format(verb, subject)

        # 根据模板类型生成答案，确保提供足够参数
        if template_idx == 0:  # 如何{verb}{subject}
            answer = a_template.format(subject, verb)
        elif template_idx == 1:  # 什么是{subject}
            # 需要3个参数: subject, 描述1, 描述2
            desc1 = random.choice(descriptors)
            desc2 = random.choice(descriptors)
            answer = a_template.format(subject, desc1, desc2)
        elif template_idx == 2:  # {subject}怎么安装
            # 需要1个参数: verb
            answer = a_template.format(verb)
        elif template_idx == 3:  # {subject}多少钱
            # 需要3个参数: 价格1, 价格2, 价格3
            price1 = random.randint(10, 100)
            price2 = random.randint(100, 1000)
            price3 = random.randint(1000, 10000)
            answer = a_template.format(price1, price2, price3)
        elif template_idx == 4:  # {subject}有什么功能
            # 需要2个参数: 功能1, 功能2
            feature1 = f"{random.choice(verbs)}功能"
            feature2 = f"{random.choice(descriptors)}功能"
            answer = a_template.format(feature1, feature2)

        qa_list.append((question, answer))

    return qa_list

if __name__ == "__main__":
    # 1. 生成示例数据
    qa_database = generate_sample_qa(5000)
    print(qa_database)

    print(f"✅ 已生成 {len(qa_database)} 条问答对")

    # 2. 初始化不同模式的系统
    vector_system = QASystem(qa_database, mode="vector")



    print("\n🏃‍♂️ 开始性能基准测试...")
    test_question = "激活会员服务的详细步骤"

    # 纯向量性能
    start = time.time()
    _vecto = vector_system.query(test_question, top_k=10)

    print(f'单纯向量下的输出结果....：{_vecto}')
    vector_time = time.time() - start





    print("\n📈 性能对比:")
    print(f"  纯向量检索: {vector_time * 1000:.2f}ms")
