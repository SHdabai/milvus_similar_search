# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
# Time:  15:44
# FileName: milvus_db
# Tools: PyCharm

import asyncio
from milvus_src.milvus_connection import MilvusConnectionPool
from typing import List, Dict, Any, Optional

from typing import List
import logging

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MilvusQA")



class DatabaseManager:
    """Milvus数据库操作管理类"""

    def __init__(self, conn_pool: MilvusConnectionPool):
        """
        初始化数据库管理器
        :param conn_pool: Milvus连接池实例
        """
        self.conn_pool = conn_pool

    async def create_database(self, db_name: str):
        """
        创建新的数据库
        :param db_name: 要创建的数据库名称

        注意：创建数据库操作需要连接到默认数据库执行
        """
        # 使用默认数据库连接
        conn = await self.conn_pool.get_connection("default")
        try:
            logger.info(f"正在创建数据库: {db_name}")
            # Milvus原生操作不需要在初始化时指定db_name
            conn.create_database(db_name=db_name)
            logger.info(f"成功创建数据库: {db_name}")
        except Exception as e:
            logger.error(f"创建数据库失败: {e}")
            raise
        finally:
            await self.conn_pool.release_connection(conn)

    async def drop_database(self, db_name: str):
        """
        删除数据库
        :param db_name: 要删除的数据库名称

        注意：删除数据库操作需要连接到默认数据库执行
        """
        # 使用默认数据库连接
        conn = await self.conn_pool.get_connection("default")
        try:
            logger.warning(f"正在删除数据库: {db_name}")
            conn.drop_database(db_name=db_name)
            logger.warning(f"已删除数据库: {db_name}")
        except Exception as e:
            logger.error(f"删除数据库失败: {e}")
            raise
        finally:
            await self.conn_pool.release_connection(conn)

    async def list_databases(self) -> List[str]:
        """
        列出所有数据库
        :return: 数据库名称列表

        注意：列出数据库操作需要连接到默认数据库执行
        """
        # 使用默认数据库连接
        conn = await self.conn_pool.get_connection("default")
        try:
            logger.info("正在列出所有数据库")
            return conn.list_databases()
        finally:
            await self.conn_pool.release_connection(conn)

    async def describe_database(self, db_name: str) -> Dict:
        """
        获取数据库的详细信息
        :param db_name: 数据库名称
        :return: 数据库的详细信息字典

        注意：描述数据库操作需要连接到目标数据库执行（新修改）
        """
        # 连接到目标数据库（此处已修改）
        conn = await self.conn_pool.get_connection(db_name)
        try:
            logger.info(f"正在获取数据库信息: {db_name}")
            # 目标数据库操作在目标库连接上执行
            return conn.describe_database(db_name=db_name)
        finally:
            await self.conn_pool.release_connection(conn)



async def demo_db_manager():
    uri = "http://localhost:31006"
    # uri = "http://localhost:31006"
    token = "root:Milvus"

    # 创建连接池
    conn_pool = MilvusConnectionPool(uri, token)
    db_manager = DatabaseManager(conn_pool)

    # 列出所有数据库
    databases = await db_manager.list_databases()
    print("Existing databases:", databases)

    # 数据库的详细信息
    # db_info = await db_manager.describe_database(db_name = "qa_database_2")
    # print("Database info:", db_info)
    # await db_manager.create_database("qa_database_2")
    # print("Database created successfully")
    # await db_manager.drop_database("qa_database_2")
    # print("Database dropped")
    await conn_pool.close_all()

    # try:
    #     # 创建新数据库
    #     await db_manager.create_database("qa_database_1")
    #     print("Database created successfully")
    #
    #     # 描述数据库
    #     db_info = await db_manager.describe_database("qa_database_1")
    #     print("Database info:", db_info)
    #
    #     # 列出所有数据库（包含新创建的）
    #     databases = await db_manager.list_databases()
    #     print("Updated databases:", databases)
    # finally:
    #     # 删除数据库（清理）
    #     await db_manager.drop_database("qa_database_1")
    #     print("Database dropped")
    #
    #     # 关闭连接池
    #     await conn_pool.close_all()

if __name__=="__main__":
    # 运行示例
    asyncio.run(demo_db_manager())




