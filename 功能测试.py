import asyncio
import httpx


async def search_query():
    url = "http://localhost:31007/search"  # 修改为你实际部署的地址
    payload = {
        "query": "白酒的起源是什么？",
        "collection_name": "qa_collection_1",
        "db_name": "qa_database_1",
        "output_fields": ["answer", "score"],
        "answer_limit": 5,
        "batch_size": 1,
        "max_length": 512
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=20.0)
            response.raise_for_status()
            data = response.json()
            print("响应结果:", data)
        except httpx.HTTPStatusError as e:
            print(f"请求失败，状态码: {e.response.status_code}, 错误信息: {e.response.text}")
        except Exception as e:
            print(f"请求异常: {str(e)}")


async def delete_milvus_collection():
    url = "http://localhost:31007/collection/delete"  # 修改为你的实际服务地址
    secret_token = "your_secret_token"  # 替换为你服务端的 SECRET_TOKEN

    payload = {
        "collection_name": "qa_collection_1",
        "db_name": "qa_database_1"
    }

    headers = {
        "x-auth-token": secret_token
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            print("响应结果:", data)
        except httpx.HTTPStatusError as e:
            print(f"请求失败，状态码: {e.response.status_code}, 错误信息: {e.response.text}")
        except Exception as e:
            print(f"请求异常: {str(e)}")


if __name__=="__main__":

    # todo：向量检索.......................
    asyncio.run(search_query())

    # todo：集合删除.......................
    asyncio.run(delete_milvus_collection())






