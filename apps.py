# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
# Time:  15:42
# FileName: apps
# Tools: PyCharm
# FastAPI 服务集成示例

import os
import json
import uvicorn
import base64
import emoji
import jieba
import re
import unicodedata
import numpy as np
from fastapi import FastAPI, HTTPException,Header,Request
from pydantic import BaseModel,field_validator
from typing import List, Dict, Any, Union,Optional
# from milvus_src.milvus_connection import MilvusConnectionPool
# from milvus_src.milvus_collection import CollectionManager
# from milvus_src.milvus_entity import CollectionOperator
# from milvus_src.milvus_db import DatabaseManager
# from concurrent.futures import ThreadPoolExecutor
from milvus_src.milvus_server import MilvusService
from mode_src.build_embedding import VectorRetriever
import logging
from dotenv import load_dotenv


current_file_path = os.path.abspath(__file__)# 获取当前文件的绝对路径
current_dir = os.path.dirname(current_file_path)# 获取当前文件所在的目录
env_path = os.path.join(current_dir, "config", ".env") #环境变量路径拼接

log_path = os.path.join(os.path.dirname(current_file_path), "./log", "post_log.log") #路径拼接
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

logger.info("loaded models........................!")
model_path = os.path.join(current_dir, "../BAAI", "bge-large-zh-v1.5") #路径拼接

app = FastAPI()

# 全局实例 (进程内唯一)
build_retriever = VectorRetriever(model_name=model_path)

load_dotenv(env_path)# 加载 .env 文件
SECRET_TOKEN = os.getenv("SECRET_TOKEN")  # 后端生成公钥文件密码
milvus_service_url = os.getenv("milvus_service_url")
milvus_service_token = os.getenv("milvus_service_token")
app_host = os.getenv("app_host") #检索服务 ip
app_port = os.getenv("app_port") #检索服务 port


# 创建服务
milvus_service = MilvusService(
    uri = milvus_service_url,
    token = milvus_service_token) #milvus的服务代码

milvus_coll_manager = milvus_service.get_collection_manager() # milvus的collection的增删改查
milvus_operator = milvus_service.get_collection_operator() # milvus的实体功能




class MilvusRequest(BaseModel):
    """参数模型，使用分组优化字段使用"""
    # 核心参数组
    collection_name: str = "" # milvus 集合名字
    db_name: str = "default" # milvus 数据库名字

    # 实体操作参数组
    entity_data: Optional[List[Dict[str, Any]]] = None
    entity_id: Optional[List[Union[str, int]]] = None  # 实体数据的 id 主键值
    overwrite: bool = True  #数据插入的时候是否 进行 覆盖

    # 搜索参数组
    query: Optional[List[str]] = None # 用户问题
    query_list: Optional[List[str]] = None # 用户问题 批量数据
    answer_limit: int = 5 # 检索出的结果内容数量，默认是5
    output_fields: List[str] = ["question_text", "answer"] # 检索结果输出的内容字段

    # 创建参数组
    vector_dim: Optional[int] = 1024 # 创建的collection集合的向量维度
    max_query_length: Optional[int] = 256 # 用户提问内容 默认 256
    max_answer_length: Optional[int] = 1024 #用户回答内容长度 默认 1024

    # 批量处理参数
    batch_size: int = 32
    max_length: int = 256

    # 字段验证
    @field_validator("collection_name", mode="before")
    def validate_collection_name(cls, value):
        if not value:
            raise ValueError("collection_name is required")
        return value

    @field_validator("entity_id", "entity_data", "query", "query_list", mode="before")
    def validate_list_fields(cls, value):
        if value == []:  # 允许空列表
            return value
        return value or None


def handle_error(message, status_code=-1):
    """统一错误处理"""
    logger.error(f"API Error: {message}")
    content = json.dumps({
        "status": "error",
        "code": status_code,
        "message": message
    }, ensure_ascii=False)
    raise HTTPException(status_code=status_code, detail=content)


def encode_vector_base64(vec: np.ndarray) -> str:
    return base64.b64encode(vec.tobytes()).decode("utf-8")



@app.post("/insert")
async def insert_entity(request: MilvusRequest):
    """milvus实体数据插入"""
    try:
        if not request.entity_data:
            handle_error("entity_data is required", -1)

        # for i, item in enumerate(request.entity_data):
        #     question_text = item.get("question_text", "")
        #     answer = item.get("answer", "")

            # # 校验文本长度
            # if len(question_text.encode("utf-8")) > 256:
            #     logger.warning(f"[第{i + 1}条] question_text 超过长度限制：{len(question_text)}")
            #     return json.dumps({
            #         "status": "fail",
            #         "code": -1,
            #         "data": f"❌ 第{i + 1}条数据 question_text 长度超过256字符",
            #     }, ensure_ascii=False)
            #
            # if len(answer.encode("utf-8")) > 1024:
            #     logger.warning(f"[第{i + 1}条] answer 超过长度限制：{len(answer)}")
            #     return json.dumps({
            #         "status": "fail",
            #         "code": -1,
            #         "data": f"❌ 第{i + 1}条数据 answer 长度超过1024字符",
            #     }, ensure_ascii=False)
            #
            # # 兼容 question_vector / embedding_vector 字段
            # vector = item.get("question_vector")
            # if not vector or len(vector) != 1024:
            #     logger.warning(f"[第{i + 1}条] 向量维度异常，当前维度：{len(vector) if vector else '无'}")
            #     return json.dumps({
            #         "status": "fail",
            #         "code": -1,
            #         "data": f"❌ 第{i + 1}条数据向量维度错误，应为1024维，当前为{len(vector) if vector else '无'}",
            #     }, ensure_ascii=False)

        insert_ = await milvus_operator._insert(
            request.collection_name,
            request.entity_data,
            request.db_name,
            request.overwrite,
        )
        if not insert_ or insert_ is None:
            return json.dumps({
                "status": "success",
                "code": -1,
                "data": "✅ milvus实体数插入失败",
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "success",
                "code": 0,
                "data": "✅ milvus实体数插入成功",
            }, ensure_ascii=False)

    except Exception as e:
        handle_error(f"插入失败: {str(e)}", -1)


@app.post("/upsert")
async def upsert_entity(request: MilvusRequest):
    """milvus实体数据更新"""
    try:
        if not request.entity_data:
            handle_error("entity_data is required", -1)

        await milvus_operator._upsert(
            request.collection_name,
            request.entity_data,
            request.db_name
        )
        return json.dumps({
            "status": "success",
            "code": 0,
            "data": "✅ milvus实体数据更新成功",
        }, ensure_ascii=False)
    except Exception as e:
        handle_error(f"更新失败: {str(e)}", -1)


@app.post("/delete")
async def delete_entity(request: MilvusRequest,x_auth_token: str = Header(...)):
    """milvus实体数据删除"""
    try:
        # Token 验证
        if x_auth_token != SECRET_TOKEN:
            raise HTTPException(status_code=401, detail="🚫 权限校验失败，无权访问该接口")

        if not request.entity_id:
            handle_error("entity_id is required", -1)

        delete_ = await milvus_operator._delete(
            collection_name=request.collection_name,
            ids=request.entity_id,
            db_name=request.db_name
        )
        if not delete_:
            return json.dumps({
                "status": "success",
                "code": -1,
                "data": f"✅ 实体数据{request.entity_id}删除失败！！！！",
            }, ensure_ascii=False)

        else:
            return json.dumps({
                "status": "success",
                "code": 0,
                "data": f"✅ 实体数据{request.entity_id}删除成功......",
            }, ensure_ascii=False)
    except Exception as e:
        handle_error(f"删除失败: {str(e)}", -1)


def is_valid_semantic_text(text: str, min_length: int = 3) -> bool:
    """
    判断文本是否是有语义的中文句子，过滤掉纯表情、标点、语音提示、乱码、无意义字符等情况。
    """

    # 预处理
    if not text or not isinstance(text, str):
        return False

    text = text.strip()

    # 1. 过滤空/空格/极短字符
    if not text or len(text) < min_length:
        return False

    # 2. 过滤表情符号
    if emoji.emoji_count(text) > 0:
        return False

    # 3. 过滤纯标点（中英文）
    if all(unicodedata.category(char).startswith("P") for char in text):
        return False

    # 4. 过滤系统提示符（如 [图片]、[语音]）
    if re.fullmatch(r"\[[^\]]+\]", text):
        return False

    # 5. 过滤网址/图片链接
    if re.search(r"(http|https)://[^\s]+(\.jpg|\.png|\.gif|\.webp)?", text.lower()):
        return False

    # 6. 过滤电话号 / 纯数字
    if re.fullmatch(r"\d{6,}", text.replace("-", "").replace(" ", "")):
        return False

    # 7. 过滤拼音串（简易）或纯英文
    if re.fullmatch(r"[a-zA-Z]{3,}", text):
        return False

    # 8. 过滤重复字符堆叠（如 "啊啊啊啊啊啊啊啊"）
    if len(set(text)) <= 2 and len(text) > 6:
        return False

    # 9. 判断是否包含中文
    if not re.search(r'[\u4e00-\u9fff]', text):
        return False

    # 10. 分词后判断是否有实词（如名词、动词）
    words = list(jieba.cut(text))
    if len(words) < 2:
        return False

    return True


@app.post("/search")
async def search_entity(request: MilvusRequest):
    """搜索相似问题"""
    try:
        if not request.query:
            handle_error("query is required", -1)

        # 确保build_retriever可用
        if 'build_retriever' not in globals():
            handle_error("Embedding服务未初始化", -1)

        if not is_valid_semantic_text(request.query[0],3):
            return json.dumps({
                "status": "success",
                "code": -1,
                "data": "未能识别你的信息，请重新输入.....",
            }, ensure_ascii=False)

        describe_collection = await build_retriever.query_embedding(
            query=request.query,
            batch_size=request.batch_size,
            max_length=request.max_length,
        )

        results = await milvus_operator._search(
            collection_name=request.collection_name,
            data=describe_collection,
            limit=request.answer_limit,
            output_fields=request.output_fields,
            db_name=request.db_name,
        )
        if not results or results is None:
            return json.dumps({
                "status": "success",
                "code": -1,
                "data": "实体数据查询失败",
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "success",
                "code": 0,
                "data": results,
            }, ensure_ascii=False)

    except Exception as e:
        handle_error(f"搜索失败: {str(e)}", -1)

@app.post("/count")
async def count_entity(request: MilvusRequest):
    """milvus实体数据数量查询"""
    try:

        count_ = await milvus_operator._count(
            collection_name=request.collection_name,
            db_name=request.db_name
        )
        if count_ == -1:
            return json.dumps({
                "status": "success",
                "code": -1,
                "data": f"✅ 实体数据数量统计失败！！！！",
            }, ensure_ascii=False)

        else:
            return json.dumps({
                "status": "success",
                "code": 0,
                "data": count_,
            }, ensure_ascii=False)
    except Exception as e:
        handle_error(f"实体数据数量统计失败: {str(e)}", -1)






@app.post("/collection/delete")
async def delete_collection(request: MilvusRequest,x_auth_token: str = Header(...)):
    """milvus集合删除"""
    try:

        # Token 验证
        if x_auth_token != SECRET_TOKEN:
            raise HTTPException(status_code=401, detail="🚫 权限校验失败，无权访问该接口")

        delete_collection_ = await milvus_coll_manager.drop_collection(
            collection_name=request.collection_name,
            db_name=request.db_name
        )
        if delete_collection_ == -1:
            return json.dumps({
                "status": "success",
                "code": -1,
                "data": f"✅ 集合{request.collection_name}删除失败！！！！",
            }, ensure_ascii=False)

        else:
            return json.dumps({
                "status": "success",
                "code": 0,
                "data": f"✅ 集合{request.collection_name}删除成功.......",
            }, ensure_ascii=False)
    except Exception as e:
        handle_error(f"集合删除失败: {str(e)}", -1)


@app.post("/collection/create")
async def create_collection(request: MilvusRequest):
    """创建milvus集合"""
    try:
        if not request.vector_dim:
            handle_error("vector_dim is required", -1)

        create_collection_ = await milvus_coll_manager.create_collection(
            collection_name=request.collection_name,
            db_name=request.db_name,
            vector_dim=request.vector_dim,
            max_query_length=request.max_query_length,
            max_answer_length = request.max_answer_length
        )
        if create_collection_ == -1:
            return json.dumps({
                "status": "success",
                "code": -1,
                "data": f"✅ 集合{request.collection_name}创建失败！！！！！",
            }, ensure_ascii=False)

        else:

            return json.dumps({
                "status": "success",
                "code": 0,
                "data": f"✅ 集合{request.collection_name}创建成功.......",
            }, ensure_ascii=False)
    except Exception as e:
        handle_error(f"集合创建失败: {str(e)}", -1)


@app.post("/collection/list")
async def list_collection(request: MilvusRequest):
    """查询集合列表"""
    try:
        describe_collection = await milvus_coll_manager.list_collections(
            db_name=request.db_name,
        )

        if not describe_collection or describe_collection is None:
            return json.dumps({
                "status": "success",
                "code": -1,
                "data": f"✅ 当前集合列表: {describe_collection} 为空，错误查询",
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "success",
                "code": 0,
                "data": f"✅ 当前集合列表: {describe_collection}",
            }, ensure_ascii=False)
    except Exception as e:
        handle_error(f"集合查询失败: {str(e)}", -1)



@app.post("/query/embedding")
async def _query_embedding(request: MilvusRequest):
    """文本转向量"""
    try:
        if not request.query_list:
            handle_error("query_list is required", -1)

        # 确保build_retriever可用
        if 'build_retriever' not in globals():
            handle_error("Embedding服务未初始化", -1)

        describe_collection = await build_retriever.query_embedding(
            query=request.query_list,
            batch_size=request.batch_size,
            max_length=request.max_length,
        )

        # 保险起见检查一下类型（不做转换）
        if not isinstance(describe_collection, np.ndarray):
            handle_error("模型输出必须是 numpy ndarray", -1)
        if describe_collection.dtype != np.float32:
            handle_error("模型输出必须是 float32 类型", -1)

        def is_empty(arr: np.ndarray) -> bool:
            return arr.size == 0  # 适用于所有维度的数组

        if is_empty(describe_collection) is True:
            return json.dumps({
                "status": "success",
                "code": -1,
                "data": "向量转换数据失败",
            }, ensure_ascii=False)

        else:
            # base64 编码
            vectors_base64 = [encode_vector_base64(vec) for vec in describe_collection]


            return json.dumps({
                "status": "success",
                "code": 0,
                "data": vectors_base64,
            }, ensure_ascii=False)
            # return describe_collection
    except Exception as e:
        handle_error(f"向量转换失败: {str(e)}", -1)


if __name__ == "__main__":
    # 这些变量需要在实际部署时配置
    _host = app_host
    _port = int(app_port)
    _certfile = None
    _keyfile = None


    uvicorn.run(
        app,
        host=_host,
        port=_port,
        ssl_keyfile=_keyfile,
        ssl_certfile=_certfile
    )




