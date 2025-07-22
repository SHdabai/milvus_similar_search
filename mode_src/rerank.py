# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
# Time:  16:25
# FileName: rerank
# Tools: PyCharm





class RerankRetriever:
    '''
    RerankRetriever (精排检索器)
        使用 LightWeightFlagLLMReranker 进行细粒度匹配
        计算 query 与完整问答对的语义相关性
        精度更高但耗时较长

    '''



    def __init__(self, model_name: str = rerank_path):
        """
        初始化Rerank检索器
        :param model_name: Rerank模型名称
        """
        logging.info(f"🛠️ 正在加载Rerank模型 : {model_name}...")
        self.model = FlagReranker(
            model_name,
            use_fp16=True,  #
            device=["0"],  # 使用GPU
            cache_dir = "./rerank_cache",  # 添加缓存目录避免内存泄漏
            pool_processes = 0  # 关键：禁用多进程
        )
        self.qa_texts = []
        logging.info("✅ Rerank模型加载完成.....")

    def build_index(self, qa_pairs: List[Tuple[str, str]]):
        """
        准备问答对数据
        :param qa_pairs: 问答对列表
        """
        if not qa_pairs:
            raise ValueError("知识库不能为空！")

        logging.info(f"📋 准备 {len(qa_pairs)} 条问答对用于Rerank...")
        # 将问答对转换为文本："问题 [SEP] 答案"
        self.qa_texts = [f"{q} [SEP] {a}" for q, a in qa_pairs]

        logging.info("✅ 数据准备完成.....")
        print(f"✅ rerank数据准备完成....self.qa_texts数据内容：{self.qa_texts}")



    def search(self, query: str, top_k: int = 5, batch_size: int = 32) -> List[Tuple[float, Tuple[str, str]]]:
        """
        执行Rerank检索
        :param query: 用户查询
        :param top_k: 返回的结果数量
        :param batch_size: 批处理大小
        :return: 包含(分数, (问题, 答案))的列表
        """
        if not self.qa_texts:
            raise RuntimeError("请先调用 build_index 准备数据！")

        logging.info(f"🔍 正在对 {len(self.qa_texts)} 条候选进行Rerank...")
        scores = []
        total = len(self.qa_texts)

        # 批量处理以提高效率
        for i in tqdm(range(0, total, batch_size)):
            batch_texts = self.qa_texts[i:i + batch_size]
            # 准备查询-文本对
            pairs = [(query, text) for text in batch_texts]

            # # 计算相关性分数（优化参数）
            # batch_scores = self.model.compute_score(
            #     pairs,
            #     cutoff_layers=[28],  # 使用顶层表示
            #     compress_ratio=1.5,  # 压缩率平衡精度/速度
            #     compress_layer=[24, 40]  # 压缩层设置
            # )
            # 计算相关性分数（优化参数）
            batch_scores = self.model.compute_score(
                pairs,
                normalize=True
            )
            scores.extend(batch_scores)

        # 获取TopK结果
        '''
        np.argsort(scores)：输出 索引根据 scores的升序排列
        [::-1]:反转排序，
        '''
        indices = np.argsort(scores)[::-1][:top_k]
        results = [(float(scores[i]), self._split_qa(self.qa_texts[i])) for i in indices]



        return sorted(results, key=lambda x: x[0], reverse=True)

    def _split_qa(self, text: str) -> Tuple[str, str]:
        """将组合文本分割回问答对"""
        if " [SEP] " in text:
            return tuple(text.split(" [SEP] ", 1))
        return text, ""

    def close(self):
        """安全关闭资源"""
        try:
            # 手动释放模型资源
            if hasattr(self.model, "model"):
                del self.model.model
            if hasattr(self.model, "tokenizer"):
                del self.model.tokenizer
            # 显式垃圾回收
            gc.collect()
        except:
            pass
