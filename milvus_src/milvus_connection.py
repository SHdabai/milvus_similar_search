# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
# Time:  15:47
# FileName: milvus_connection
# Tools: PyCharm
import asyncio
from pymilvus import MilvusClient

import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MilvusQA")


class MilvusConnectionPool:
    """Milvus连接池管理类"""
    _pools = {}  # 存储不同数据库的连接池，key为db_name，value为连接池信息
    _lock = asyncio.Lock()  # 全局锁，确保线程安全

    def __init__(self, uri: str, token: str, max_workers: int = 20):
        """
        初始化连接池参数
        :param uri: Milvus服务地址
        :param token: 认证令牌
        :param max_workers: 每个数据库的最大连接数
        """
        self.uri = uri
        self.token = token
        self.max_workers = max_workers

    async def get_connection(self, db_name: str = "default") -> MilvusClient:
        """
        获取指定数据库的连接
        :param db_name: 数据库名称，默认为'default'
        :return: MilvusClient连接实例
        """
        async with self._lock:  # 加锁确保线程安全
            # 如果该数据库的连接池不存在，创建新的连接池
            if db_name not in self._pools:
                self._pools[db_name] = {
                    'pool': [],  # 连接对象列表
                    'in_use': set(),  # 正在使用的连接ID集合
                    'semaphore': asyncio.Semaphore(self.max_workers)  # 限制最大连接数
                }

            pool_data = self._pools[db_name]

            # 寻找空闲连接
            for conn in pool_data['pool']:
                if id(conn) not in pool_data['in_use']:
                    pool_data['in_use'].add(id(conn))
                    logger.info(f"获取[{db_name}]数据库的空闲连接")
                    return conn

            # 没有空闲连接则创建新连接
            async with pool_data['semaphore']:
                # 仅在非默认数据库操作时传递db_name
                client_args = {
                    "uri": self.uri,
                    "token": self.token
                }
                if db_name != "default":
                    client_args["db_name"] = db_name

                new_conn = MilvusClient(**client_args)

                # 添加自定义标记（原始MilvusClient没有db_name属性）
                new_conn._db_name = db_name
                pool_data['pool'].append(new_conn)
                pool_data['in_use'].add(id(new_conn))
                logger.info(f"创建[{db_name}]数据库的新连接，当前连接数: {len(pool_data['pool'])}")
                return new_conn

    async def release_connection(self, conn: MilvusClient):
        """
        释放连接回连接池
        :param conn: 要释放的MilvusClient连接
        """
        # 获取连接关联的数据库名（通过自定义的_db_name属性）
        db_name = getattr(conn, '_db_name', 'default')

        async with self._lock:  # 加锁确保线程安全
            if db_name in self._pools:
                conn_id = id(conn)
                pool_data = self._pools[db_name]

                # 如果连接正在使用中，则释放
                if conn_id in pool_data['in_use']:
                    pool_data['in_use'].remove(conn_id)
                    logger.info(f"释放[{db_name}]数据库的连接")
                else:
                    logger.warning(f"尝试释放未使用的连接: {conn_id}")
            else:
                logger.error(f"尝试释放不存在的数据库连接: {db_name}")

    async def close_all(self):
        """
        关闭所有连接并清理连接池
        """
        async with self._lock:  # 加锁确保线程安全
            for db_name, pool_data in self._pools.items():
                for conn in pool_data['pool']:
                    try:
                        conn.close()
                        logger.info(f"已关闭[{db_name}]数据库的连接")
                    except Exception as e:
                        logger.error(f"关闭连接时出错: {e}")

                # 清空连接池
                pool_data['pool'].clear()
                pool_data['in_use'].clear()

            # 清空连接池字典
            self._pools = {}
            logger.info("所有数据库连接已关闭并清理")