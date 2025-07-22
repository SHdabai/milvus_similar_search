# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
# Time:  10:27
# FileName: milvus_server
# Tools: PyCharm

import numpy as np
import random
import logging
from milvus_src.milvus_connection import MilvusConnectionPool
from milvus_src.milvus_collection import CollectionManager
from milvus_src.milvus_entity import CollectionOperator
from milvus_src.milvus_db import DatabaseManager
from concurrent.futures import ThreadPoolExecutor
import asyncio
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MilvusQA")

class MilvusService:
    def __init__(self, uri: str, token: str, max_workers: int = 100):
        # 连接池
        self.conn_pool = MilvusConnectionPool(uri, token, max_workers)
        # 线程池执行器
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # 初始化管理器
        self.db_manager = DatabaseManager(self.conn_pool)
        self.collection_manager = CollectionManager(self.conn_pool)
        # 复用Operator实例（避免频繁实例化）
        self.collection_operator = CollectionOperator(
            self.collection_manager,
            self.conn_pool,
            self.executor
        )

    def get_db_manager(self) -> DatabaseManager:
        return self.db_manager

    def get_collection_manager(self) -> CollectionManager:
        return self.collection_manager

    def get_collection_operator(self) -> CollectionOperator:
        return self.collection_operator

    async def close(self):
        await self.conn_pool.close_all()
        self.executor.shutdown()


async def demo_collection_operator():
    uri = "http://localhost:31006"
    token = "root:Milvus"

    # 创建线程池
    executor = ThreadPoolExecutor(max_workers=50)

    from mode_src.build_embedding import VectorRetriever
    import os
    current_file_path = os.path.abspath(__file__)  # 获取当前文件的绝对路径
    current_dir = os.path.dirname(current_file_path)  # 获取当前文件所在的目录
    model_path = os.path.join(current_dir, "../../BAAI", "bge-large-zh-v1.5")  # 路径拼接
    build_retriever = VectorRetriever(model_name=model_path)

    # 创建服务
    service = MilvusService(uri, token)
    coll_manager = service.get_collection_manager()
    operator = service.get_collection_operator()

    db_name = "qa_database_2"
    collection_name = "demo_qa_collection"

    try:
        # 创建并加载集合
        # await coll_manager.create_collection(
        #     collection_name=collection_name,
        #     db_name=db_name,
        #     vector_dim=768
        # )
        await coll_manager.load_collection(collection_name, db_name)


        # 准备数据
        questions = [
            "What is Python?",
            "答案：作为有七十余年发展史的老酒厂，碧春酒厂在茅台酒酿造工艺基础上，不断对酿酒工艺深入研究、改进，细化了30道工序和165个工艺，将高效、稳定的酿酒技术运用于基酒生产的每一个环节，并斩获国家「一种新型酱香酒酿造方法」的发明专利；在酒曲制上碧春酒厂引入茅台酒发酵物中的16株优秀种菌，不断丰富、完善菌种族群，成功研发保密酒曲，并被列为「商业秘密保护单位」。酿造工艺和酒曲技术的升级，让碧春酒拥有了更为细腻、柔和的风味物质，入口顺滑，酱香醇厚，余味悠长。",
            "What are Python decorators?"
        ]
        answers = [
            "Python is a popular programming language",
            "答案：作为有七十余年发展史的老酒厂，碧春酒厂在茅台酒酿造工艺基础上，不断对酿酒工艺深入研究、改进，细化了30道工序和165个工艺，将高效、稳定的酿酒技术运用于基酒生产的每一个环节，并斩获国家「一种新型酱香酒酿造方法」的发明专利；在酒曲制上碧春酒厂引入茅台酒发酵物中的16株优秀种菌，不断丰富、完善菌种族群，成功研发保密酒曲，并被列为「商业秘密保护单位」。酿造工艺和酒曲技术的升级，让碧春酒拥有了更为细腻、柔和的风味物质，入口顺滑，酱香醇厚，余味悠长。",
            "Decorators are functions that modify other functions"
        ]

        # 生成向量示例（真实应用中会用模型生成）
        # vectors = [
        #     [random.random().tolist() for _ in range(1024)],
        #     [random.random().tolist() for _ in range(1024)],
        #     [random.random().tolist() for _ in range(1024)]
        # ]
        vectors = await build_retriever.query_embedding(
                query=questions,
                batch_size=32,
                max_length=64,
            )

        # # 插入数据
        data = [
            # {"id": 4, "question_vector": vectors[0], "question_text": questions[0], "answer": answers[0]},
            # {"id": 2, "question_vector": vectors[1], "question_text": questions[1], "answer": answers[1]},
            # {"id": 3, "question_vector": vectors[2], "question_text": questions[2], "answer": answers[2]},
            # {"id": 4, "question_vector": vectors[2], "question_text": questions[2], "answer": answers[2]},
            {"id": 5, "question_vector": vectors[2], "question_text": questions[2], "answer": answers[2]},
            {"id": 6, "question_vector": vectors[2], "question_text": questions[2], "answer": answers[2]},
            {"id": 7, "question_vector": vectors[2], "question_text": questions[2], "answer": answers[2]}
        ]
        #
        inserted_ids = await operator._insert(collection_name, data, db_name)
        print(f"Inserted IDs: {inserted_ids}")


        #
        # 更新答案
        # update_data = [
        #     # {"id": 2, "answer": "Visit python.org/downloads and run the installer"}
        #     {"id": 5, "question_vector": vectors[1], "question_text": questions[1], "answer":"989876579868"}
        # ]
        # await operator.upsert(collection_name, update_data, db_name)
        # print("Answer updated")



        # # 删除一条记录
        # await operator._delete(
        #     collection_name=collection_name,
        #     ids=[5],
        #     db_name=db_name
        # )
        # print("Deleted record with ID 5")
        # #
        # # # # 获取数据量
        # counts = await operator._count(collection_name,db_name)
        # print(f"Collection contains {counts} entities")

        # 查询特定数据
        # results = await operator.query(collection_name,"user_id == 12345",["id","question_text","answer"],db_name,[1,2,3])
        # results = await operator.query(collection_name,"id == 5",["id","question_text","answer"],db_name)
        # print(f"Query results：{results} ")

        # 搜索相似问题
        # query_vector = [v * 1.05 for v in vectors[0]]  # 轻微修改的查询向量
        # describe_collection = await build_retriever.query_embedding(
        #     query=['九牛信息咋样'],
        #     batch_size=32,
        #     max_length=64,
        # )
        # print("EEEEEEEEE:",describe_collection)
        # print("EEEEEEEEE:",type(describe_collection))

        #  # inserted_ids = await operator._insert(collection_name, data, db_name)
        # print(f"Inserted IDs: {inserted_ids}")


        # results = await operator._search(
        #     collection_name=collection_name,
        #     # data=[query_vector],
        #     data=describe_collection ,
        #     limit=2,
        #     output_fields=["id","question_text", "answer"],
        #     db_name=db_name,
        # )
        # #
        # print(f'milvus 查询的结果内容...........：{results}')
        # from pprint import pprint
        # pprint(results)
        '''
        输出内容格式：
        [
            {
                "question_text": "如何重置密码？",
                "answer": "访问设置页面点击重置链接",
                "similarity": 0.92
            },
            {
                "question_text": "忘记密码怎么办？",
                "answer": "通过邮箱接收重置链接",
                "similarity": 0.87
            }
        ]
        '''



    finally:
    #     # 清理资源
        await service.close()
        # pass
if __name__=="__main__":
    # 运行示例
    asyncio.run(demo_collection_operator())