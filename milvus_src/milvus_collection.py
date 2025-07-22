# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
# Time:  15:45
# FileName: milvus_collection
# Tools: PyCharm


import asyncio
from pymilvus import MilvusClient, DataType, utility
from typing import List, Dict, Any, Optional

import logging
from milvus_src.milvus_connection import MilvusConnectionPool




# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MilvusQA")


class CollectionManager:
    def __init__(self, conn_pool: MilvusConnectionPool):
        self.conn_pool = conn_pool
        self.loaded_collections = {}
        self.load_locks = {}

    async def _acquire_load_lock(self, db_name: str, collection_name: str):
        key = f"{db_name}:{collection_name}"
        if key not in self.load_locks:
            self.load_locks[key] = asyncio.Lock()
        return self.load_locks[key]

    async def create_collection(
            self,
            collection_name: str,
            db_name: str = "default",
            vector_dim: int = 1024,
            max_query_length: int = 256,
            max_answer_length: int = 1024
    ):
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True,auto_id=False )
        schema.add_field(field_name="question_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field(field_name="question_text", datatype=DataType.VARCHAR, max_length=max_query_length)
        schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=max_answer_length)

        conn = await self.conn_pool.get_connection(db_name)

        try:
            conn.create_collection(
                collection_name=collection_name,
                schema=schema,
            )
        except Exception as e:
            return -1
        finally:
            await self.conn_pool.release_connection(conn)

        # 创建索引
        index_params = conn.prepare_index_params()
        index_params.add_index(field_name="question_vector", index_type="AUTOINDEX", metric_type="COSINE")

        conn.create_index(
            collection_name=collection_name,
            index_params=index_params
        )



    async def drop_collection(self, collection_name: str, db_name: str = "default"):
        conn = await self.conn_pool.get_connection(db_name)
        try:
            conn.drop_collection(collection_name=collection_name)

        except Exception as e:
            return -1
        finally:
            await self.conn_pool.release_connection(conn)

        # 清理加载状态
        key = f"{db_name}:{collection_name}"
        if key in self.loaded_collections:
            del self.loaded_collections[key]

    async def list_collections(self, db_name: str = "default") -> List[str]:
        conn = await self.conn_pool.get_connection(db_name)
        try:
            return conn.list_collections()
        except Exception as e:
            return []

        finally:
            await self.conn_pool.release_connection(conn)


    async def _describe_collection(self,collection_name: str, db_name: str = "default") -> List[str]:
        conn = await self.conn_pool.get_connection(db_name)
        try:
            return conn.describe_collection(collection_name=collection_name)
        finally:
            await self.conn_pool.release_connection(conn)



    async def rename_collection(
            self,
            old_name: str,
            new_name: str,
            db_name: str = "default"
    ):
        conn = await self.conn_pool.get_connection(db_name)
        try:
            conn.rename_collection(old_name=old_name, new_name=new_name)
        finally:
            await self.conn_pool.release_connection(conn)

        # 更新加载状态
        old_key = f"{db_name}:{old_name}"
        if old_key in self.loaded_collections:
            self.loaded_collections[f"{db_name}:{new_name}"] = self.loaded_collections.pop(old_key)

    async def load_collection(
            self,
            collection_name: str,
            db_name: str = "default",
            force: bool = False
    ):
        lock = await self._acquire_load_lock(db_name, collection_name)
        async with lock:
            key = f"{db_name}:{collection_name}"

            if not force and self.loaded_collections.get(key, False):
                return True

            conn = await self.conn_pool.get_connection(db_name)
            try:
                conn.load_collection(collection_name=collection_name)
                # 检查加载状态
                state = conn.get_load_state(collection_name=collection_name)
                if str(state['state']) == "Loaded":
                    self.loaded_collections[key] = True
                    return True
                return False
            finally:
                await self.conn_pool.release_connection(conn)

    async def release_collection(
            self,
            collection_name: str,
            db_name: str = "default"
    ):
        lock = await self._acquire_load_lock(db_name, collection_name)
        async with lock:
            key = f"{db_name}:{collection_name}"
            conn = await self.conn_pool.get_connection(db_name)
            try:
                conn.release_collection(collection_name=collection_name)
                self.loaded_collections.pop(key, None)
                return True
            finally:
                await self.conn_pool.release_connection(conn)

    async def is_collection_loaded(
            self,
            collection_name: str,
            db_name: str = "default"
    ) -> bool:
        key = f"{db_name}:{collection_name}"
        return self.loaded_collections.get(key, False)



async def demo_collection_manager():
    uri = "http://localhost:31006"
    token = "root:Milvus"

    # 创建连接池
    conn_pool = MilvusConnectionPool(uri, token)
    coll_manager = CollectionManager(conn_pool)

    db_name = "qa_database_2"
    collection_name = "test_8"

    try:
        # 创建新的QA集合
        # b = await coll_manager.create_collection(
        #     collection_name=collection_name,
        #     db_name=db_name,
        #     vector_dim=1024
        # )
        # print("Collection created")
        # print(f'知识库内容：{b}')
        # 列出所有集合
        collections = await coll_manager.list_collections(db_name)
        print("Collections:", collections)

        # 删除集合
        # await coll_manager.drop_collection(collection_name, db_name)
        #
        # 加载集合
        # loaded = await coll_manager.load_collection(collection_name, db_name)
        # print(f"Collection loaded: {loaded}")
        # #
        # # 检查加载状态
        # is_loaded = await coll_manager.is_collection_loaded(collection_name, db_name)
        # print(f"Is collection loaded? {is_loaded}")

        # 查看collection 的详细信息
        # from pprint import pprint
        # describe_collection = await coll_manager._describe_collection(collection_name,db_name)
        # # print("describe_collection:", describe_collection)
        # pprint(describe_collection)


        # # 重命名集合
        # new_name = "renamed_demo_collection"
        # await coll_manager.rename_collection(collection_name, new_name, db_name)
        # print(f"Collection renamed to {new_name}")
        #
        # # 释放集合
        # released = await coll_manager.release_collection(collection_name, db_name)
        # print(f"Collection released: {released}")
    finally:
        # 删除集合（清理）
        # await coll_manager.drop_collection(new_name, db_name)
        # print("Collection dropped")
        pass
        # 关闭连接池
        # await conn_pool.close_all()

if __name__=="__main__":
    # 运行示例
    asyncio.run(demo_collection_manager())

