# milvus_similar_search
基于bge基础模型结合milvus向量数据库的相似检索任务



docker run -d \
  -p 31007:31007 \
  -v /home/bert_work/BAAI:/home/BAAI \
  -v /home/bert_work/similar_search:/home/similar_search \
  --name similar_search_container \
  similar_search:dev-1.0

| 参数                | 作用                          |
| ----------------- | --------------------------- |
| `-p 31007:31007`  | 将宿主机的 31007 映射到容器的 31007 端口 |
| `-v 本地路径:容器路径`    | 将本地的模型目录挂载到容器内同级目录中         |
| `--name`          | 给容器取个名字，方便后续管理              |
| `your_image_name` | 用你实际构建好的镜像名替代               |


#删除服务中增加一个 认证服务  Auth

curl -X POST http://localhost:8000/collection/delete \
  -H "Content-Type: application/json" \
  -H "X-Auth-Token: SuperSecretDeleteToken123!@#" \
  -d '{"collection_name": "my_collection", "db_name": "default"}'
  增加一个  X-Auth-Token  来认证，进行加密  咋样   简单易用



#------------------------启动服务-------------------------

similar_search #环境名称

命令行启动主程序不使用docker

hypercorn apps.py:app -c hypercorn.toml


nohup hypercorn apps.py:app --bind 0.0.0.0:31007 > log/server_logs.txt 2>&1 &

    > 表示 覆盖写入 日志文件；
    
    每次启动都会清空 log/server_logs.txt 里的内容，写入新的日志；
    
    ✅ 适合你每次都想从头查看服务运行情况的场景。

nohup hypercorn apps.py:app --bind 0.0.0.0:31007 >> log/server_logs.txt 2>&1 &

    >> 表示 追加写入；
    
    日志会不断追加到 log/server_logs.txt 文件末尾；
    
    ✅ 适合长期运行服务，保留所有历史日志。

✅ & 表示后台运行
✅ nohup 防止终端关闭时程序退出
✅ logs.txt 是输出日志文件，可选


#------------------------停止服务-------------------------

**查看日志信息**

tail -n 5 log/output*.log

**查看31007端口号使用情况**

sudo netstat -tulnp | grep 31007

sudo kill -9 <进程号>







