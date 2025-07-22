# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
# Time:  15:45
# FileName: milvus_entity
# Tools: PyCharm
import logging
from milvus_src.milvus_collection import CollectionManager
from milvus_src.milvus_connection import MilvusConnectionPool
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional,Union
import random

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MilvusQA")


class CollectionOperator:
    def __init__(
            self,
            collection_manager: CollectionManager,
            conn_pool: MilvusConnectionPool,
            executor: ThreadPoolExecutor
    ):
        """
        初始化集合操作器

        Args:
            collection_manager (CollectionManager): 集合管理实例
            conn_pool (MilvusConnectionPool): Milvus连接池
            executor (ThreadPoolExecutor): 线程池执行器
        """
        self.collection_manager = collection_manager
        self.conn_pool = conn_pool
        self.executor = executor
        # 状态缓存 - 跟踪已加载的集合
        self.loaded_cache = {}
        # 加载状态锁 - 防止并发加载时的竞态条件
        self.load_lock = asyncio.Lock()

    async def _ensure_loaded(
            self,
            collection_name: str,
            db_name: str = "default"
    ):
        """
        确保指定集合已加载（线程安全）

        Args:
            collection_name (str): 集合名称
            db_name (str): 数据库名称（默认"default"）
        """
        # 生成缓存键
        key = f"{db_name}:{collection_name}"

        # 第一次快速检查（无锁）
        if key in self.loaded_cache:
            return

        # 加锁保证线程安全
        async with self.load_lock:
            # 再次检查（避免在等待锁时其他线程已完成加载）
            if key in self.loaded_cache:
                return

            # 检查集合加载状态
            is_loaded = await self.collection_manager.is_collection_loaded(
                collection_name,
                db_name
            )

            # 如果未加载则执行加载
            if not is_loaded:
                await self.collection_manager.load_collection(
                    collection_name,
                    db_name
                )

            # 更新缓存
            self.loaded_cache[key] = True



    #todo: 1. milvus的实体数据插入.............
    async def _insert(
            self,
            collection_name: str,
            data: List[Dict[str, Any]],
            db_name: str = "default",
            overwrite: bool = True,
    ) -> List[int]:


        """
        改进的插入方法：支持重复ID覆盖
        新增参数：
          overwrite (bool): True = 覆盖已存在ID, False = 允许重复
        返回：
          插入成功的ID列表（实际插入数可能少于输入）
        """
        await self._ensure_loaded(collection_name, db_name)
        conn = await self.conn_pool.get_connection(db_name)
        try:
            loop = asyncio.get_running_loop()

            # 获取集合主键信息
            if ":" in collection_name:
                db_name, pure_collection = collection_name.split(":", 1)
            else:
                pure_collection = collection_name

            schema = await loop.run_in_executor(
                self.executor,
                lambda: conn.describe_collection(pure_collection)
            )

            # 提取主键字段名
            primary_key = None
            for field in schema.get("fields", []):
                if field.get("is_primary", False):
                    primary_key = field["name"]
                    break

            # 分组数据：新ID vs 已存在ID
            existing_ids = set()
            if primary_key and overwrite:
                # 查询已存在的ID (批量查询优化)
                primary_values = [item[primary_key] for item in data]
                filter_expr = f"{primary_key} in {primary_values}" if len(primary_values) > 1 else \
                    f"{primary_key} == {primary_values[0]}"

                existing_items = await self._query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=[primary_key],
                    db_name=db_name
                )
                existing_ids = {item[primary_key] for item in existing_items}

            # 分离要插入和覆盖的数据
            new_items = []
            overwrite_items = []
            overwrite_ids = []

            for item in data:
                if primary_key and primary_key in item:
                    if item[primary_key] in existing_ids and overwrite:
                        overwrite_items.append(item)
                        overwrite_ids.append(item[primary_key])
                    else:
                        new_items.append(item)
                else:
                    new_items.append(item)

            # 执行操作
            result = []

            # 插入新数据
            if new_items:
                new_result = await loop.run_in_executor(
                    self.executor,
                    conn.insert,
                    collection_name,
                    new_items
                )
                result.extend(new_result)

            # 覆盖已存在数据（使用 upsert）
            if overwrite_items:
                for item in overwrite_items:
                    overwrite_result = await loop.run_in_executor(
                        self.executor,
                        conn.upsert,
                        collection_name,
                        [item]
                    )
                    result.append(overwrite_result.get("upserted_pk", item[primary_key]))

            # 关键修复: 插入后强制刷新数据到磁盘 ★★★
            ####################################################
            if new_items or overwrite_items:
                await loop.run_in_executor(
                    self.executor,
                    conn.flush,  # 调用flush强制持久化数据
                    collection_name  # 指定刷新当前集合
                )
            ####################################################

            # print(f'插入后的结果内容............：{result}')
            return result

        except Exception as e:
            # print(f"插入失败999999999999999: {str(e)}")
            return  []
        finally:
            await self.conn_pool.release_connection(conn)



    # todo: 2. milvus的实体数据更新操作.............
    async def _upsert(
            self,
            collection_name: str,
            data: List[Dict[str, Any]],
            db_name: str = "default"
    ) -> Dict:
        """
        向集合中更新插入数据（存在则更新，不存在则插入）
        """
        await self._ensure_loaded(collection_name, db_name)
        conn = await self.conn_pool.get_connection(db_name)
        try:
            loop = asyncio.get_running_loop()

            # === 执行upsert操作 ===
            result = await loop.run_in_executor(
                self.executor,
                conn.upsert,  # 底层Milvus upsert方法
                collection_name,
                data
            )

            # === 关键：添加flush操作确保立即可见 ===
            ################################################
            # 使用与insert方法相同的刷新逻辑
            if data:  # 只在有数据时刷新
                await loop.run_in_executor(
                    self.executor,
                    conn.flush,  # 调用flush强制持久化数据
                    collection_name  # 指定刷新当前集合
                )

                # 可选：重新加载集合确保元数据更新
                if ":" in collection_name:
                    _, pure_collection = collection_name.split(":", 1)
                else:
                    pure_collection = collection_name
                await loop.run_in_executor(
                    self.executor,
                    lambda: conn.load_collection(pure_collection)
                )
            ################################################

            return result


        finally:
            await self.conn_pool.release_connection(conn)



    # todo: 3. milvus的依据 id 参数进行实体数据.............
    async def _query(
            self,
            collection_name: str,
            filter: str,
            output_fields: Optional[List[str]] = None,
            db_name: str = "default",
            ids: Optional[Union[List, str, int]] = None
    ) -> List[Dict]:
        """
        查询集合中的数据（根据属性条件）

        Args:
            collection_name (str): 集合名称
            filter (str): 查询条件表达式
            output_fields (Optional[List[str]]): 要返回的字段（None返回所有字段）
            db_name (str): 数据库名称（默认"default"）

        Returns:
            List[Dict]: 查询结果列表
        """
        await self._ensure_loaded(collection_name, db_name)
        conn = await self.conn_pool.get_connection(db_name)
        try:
            loop = asyncio.get_running_loop()
            # 在独立线程中执行查询操作
            results = await loop.run_in_executor(
                self.executor,
                conn.query,  # 底层Milvus查询方法
                collection_name,
                filter,
                output_fields,
                ids
            )
            return results

        except Exception as e:
            # print(f"插入失败: {str(e)}")
            return  []

        finally:
            await self.conn_pool.release_connection(conn)



    # todo: 4. milvus的实体数据向量检索.............
    async def _search(
            self,
            collection_name: str,
            data: List[List[float]],
            limit: int = 5,
            output_fields: Optional[List[str]] = None,
            db_name: str = "default"
    ) -> List[Dict]:
        """
        向量相似度搜索（返回标准JSON格式结果）

        Args:
            collection_name (str): 集合名称
            data (List[List[float]]): 查询向量列表
            limit (int): 返回结果数量限制
            output_fields (Optional[List[str]]): 要返回的字段（默认返回问题和答案）
            db_name (str): 数据库名称（默认"default"）

        Returns:
            List[Dict]: 搜索结果列表，格式为[{},{},...]，
                        按余弦相似度从高到低排序（相似度越高越靠前）
        """
        await self._ensure_loaded(collection_name, db_name)
        conn = await self.conn_pool.get_connection(db_name)
        try:
            if output_fields is None:
                output_fields = ["id","question_text", "answer"]

            # 配置搜索参数（使用余弦相似度）
            search_params = {
                "anns_field": "question_vector",
                "metric_type": "COSINE",
                "params": {"nprobe": 20}
            }
            '''
            metric_type的参数解读：
            1. L2：欧几里得距离（平方差距离），通常用于 图像、位置、数值距离等。
            2. IP：向量点积（用于衡量方向一致性），一般用于向量归一化后用于相似度匹配，推荐系统等。
            3. COSINE： 余弦相似度
            4. JACCARD：集合交集 / 并集比，通常用于稀疏布尔向量（例如用户点击行为）
            5. HAMMING：两个二进制向量间不相同位数，通常用于哈希编码或布尔特征
            
            IP 一般配合 归一化的向量 来表示“相似度”。
            IP 越大代表越相似，所以 Milvus 会默认将结果按距离（点积）从大到小排序，不需要你手动 1 - distance。
            L2 按 distance 从小到大 排序，因为距离越小越相似。但是输出的结果也是 越相似越靠前。
                如果你关心“方向”，用 COSINE；
                如果你关心“方向 + 权重/强度”，用 IP；
                如果你归一化了向量，其实 IP ≈ COSINE，选哪个影响不大。
                        
            '''


            loop = asyncio.get_running_loop()
            raw_results = await loop.run_in_executor(
                            self.executor,
                            conn.search,  # 底层方法
                            collection_name,  # 参数1: 集合名
                            data,  # 参数2: 查询向量 查询向量列表  [ [ random.uniform(-1, 1) for _ in range(768) ] ]
                            "",  # 参数3: filter (空字符串)
                            limit,  # 参数4: 结果数量限制  int类型数据
                            output_fields,  # 参数5: 返回字段 ["id","question_text", "answer"]
                            search_params,  # 参数6: 搜索参数
                            None,  # timeout (可选)
                            None  # partition_names (可选)
            )

            # ================== 关键修改部分 ==================
            # 1. 提取实际搜索结果（假设单个查询向量）
            if len(raw_results) == 0:
                return []

            # Milvus返回格式: [SearchResult(query_vector1), SearchResult(query_vector2)]
            search_hits = raw_results[0]  # 只处理第一个查询向量的结果

            # 2. 按相似度排序（COSINE距离越小表示相似度越高）
            # 注意：Milvus默认返回顺序已经按距离升序排列（距离越小越相似）
            # 但为确保顺序正确，显式排序
            sorted_hits = sorted(
                search_hits,
                key=lambda x: x.distance,  # 使用余弦距离
                reverse=True  # 降序排序：
            )

            # 3. 转换为标准JSON格式
            results = []
            for hit in sorted_hits:
                # 从实体中提取字段
                entity = hit.entity
                result_dict = {
                    field: entity.get(field) for field in output_fields
                }

                # 可选：添加相似度分数（实际是距离值）
                # 余弦相似度 = 1 - 余弦距离
                result_dict["similarity"] = hit.distance


                results.append(result_dict)

            return results
            # ================== 修改结束 ==================

        except Exception as e:
            # print(f"插入失败: {str(e)}")
            return  []

        finally:
            await self.conn_pool.release_connection(conn)



    # todo: 5. milvus的实体数据数量统计.............
    async def _count(
            self,
            collection_name: str,
            db_name: str = "default"
    ) -> int:
        """
        改进的实体计数方法（针对 Milvus 2.5+ 优化）
        """
        try:
            # 首先尝试直接计数方法
            direct_count = await self._get_direct_count(collection_name, db_name)
            if direct_count >= 0:
                return direct_count

            # 尝试聚合方法
            return await self.safe_count(collection_name, db_name)

        except Exception as e:
            return -1

    async def _get_direct_count(
            self,
            collection_name: str,
            db_name: str
    ) -> int:
        """尝试直接获取计数"""
        conn = await self.conn_pool.get_connection(db_name)
        try:
            loop = asyncio.get_running_loop()

            # 检查并执行新的 count API
            if hasattr(conn, "count"):
                return await loop.run_in_executor(
                    self.executor,
                    lambda: conn.count(collection_name)
                )

            # 使用描述信息获取
            desc = await loop.run_in_executor(
                self.executor,
                lambda: conn.describe_collection(collection_name)
            )
            return desc.get("num_entities", -1)

        except Exception:
            return -1
        finally:
            await self.conn_pool.release_connection(conn)

    async def safe_count(
            self,
            collection_name: str,
            db_name: str = "default"
    ) -> int:
        """
        可靠的实体计数方法（兼容所有版本）
        """
        conn = await self.conn_pool.get_connection(db_name)
        try:
            # 确保集合已加载
            await self._ensure_loaded(collection_name, db_name)

            loop = asyncio.get_running_loop()

            # 优先使用聚合查询
            try:
                # 构造聚合查询
                result = await loop.run_in_executor(
                    self.executor,
                    lambda: conn.query(
                        collection_name=collection_name,
                        filter="",  # 所有记录
                        output_fields=["count(*)"],
                        use_aggregation=True
                    )
                )

                if result and isinstance(result, list) and len(result) > 0:
                    count_value = result[0].get("count(*)", -1)
                    return int(count_value)
            except Exception as e:
                print(f"聚合查询失败: {str(e)}")

            # 回退到简单查询计数
            results = await loop.run_in_executor(
                self.executor,
                lambda: conn.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["id"],  # 只返回ID减少数据量
                    limit=1000000  # 设置足够大的上限
                )
            )

            return len(results)

        except Exception as e:
            print(f"安全计数失败: {str(e)}")
            return -1
        finally:
            await self.conn_pool.release_connection(conn)





    async def _delete(
            self,
            collection_name: str,
            ids: Optional[Union[list, str, int]] = None,
            db_name: str = "default",
    ) -> Dict:
        """
        从集合中删除数据（增强版）
        """
        await self._ensure_loaded(collection_name, db_name)
        conn = await self.conn_pool.get_connection(db_name)
        try:
            loop = asyncio.get_running_loop()

            # 获取纯集合名称（移除数据库前缀）
            if ":" in collection_name:
                _, pure_collection = collection_name.split(":", 1)
            else:
                pure_collection = collection_name

            # === 关键修改：在删除操作前加载集合 ===
            await loop.run_in_executor(
                self.executor,
                lambda: conn.load_collection(pure_collection)
            )

            # === 执行删除操作 ===
            result = await loop.run_in_executor(
                self.executor,
                conn.delete,
                pure_collection,  # 使用纯集合名
                ids
            )

            # === 关键修复：添加与insert方法一致的刷新逻辑 ===
            ####################################################
            # 使用与insert方法完全相同的刷新方式
            # 注意：使用相同的刷新调用方式和参数
            await loop.run_in_executor(
                self.executor,
                conn.flush,  # 使用与insert完全相同的flush方式
                collection_name  # 指定刷新当前集合
            )
            ####################################################

            # === 再次加载确保状态更新 ===
            await loop.run_in_executor(
                self.executor,
                lambda: conn.load_collection(pure_collection)
            )
            # print(f'删除后的结果内容............：{result}')
            return result

        except Exception as e:
            # print(f"插入失败: {str(e)}")
            return {}
        finally:
            await self.conn_pool.release_connection(conn)

