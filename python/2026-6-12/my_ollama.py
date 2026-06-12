import chromadb
import pymysql
import os
import hashlib
import datetime
import re
import json

from ollama import Client
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from collections import defaultdict
from FlagEmbedding import FlagReranker
from llama_cpp import Llama

class FU01:
    def __init__(self):
        # ollama
        self.ollama_client = Client(
            host='http://localhost:11434',
            headers={'x-some-header': 'some-value'}
            )
        
        # chromadb
        chroma_client = chromadb.PersistentClient(path="G:\\chromadb")
        self.chroma_collection = chroma_client.get_or_create_collection(name="my_data")

        # Elasticsearch
        self.es_client = Elasticsearch(hosts="http://localhost:9200")

        # Elasticsearch 初始化 
        if not self.es_client.indices.exists(index="my_index"):
            self.es_client.indices.create(index="my_index", mappings= {
                "properties": {
                    "content": {"type": "text", 
                                "analyzer": "ik_max_word", 
                                "search_analyzer": "ik_smart"},
                    "title": {"type": "text", 
                                "analyzer": "ik_max_word", 
                                "search_analyzer": "ik_smart"},
                    "mysql_id": {"type": "long"},
                    "mysql_table": {"type": "keyword"},
                    "start_index": {"type": "integer"},
                    "end_index": {"type": "integer"}
                }
            })

        self.reranker = FlagReranker(r"G:\img_WD14\bge-reranker-v2-m3", use_fp16=False, devices='cpu') 

        # Llama 
        self.llm = Llama(
            model_path=r"G:\img_WD14\GGUF\Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q4_K.gguf",
            n_ctx=32*1024,
            n_gpu_layers=10,
            flash_attn=True, #  Use flash attention.
            use_mmap=True, #  Use mmap if possible.
            offload_kqv=True, # Offload K, Q, V to GPU.
            chat_format="chatml"
            )
        
        self.llm_embedding = Llama(
            model_path=r"G:\img_WD14\GGUF\Qwen3-Embedding-8B-Q4_K_M.gguf", embedding=True,
            n_ctx=2*1024,
            n_gpu_layers=0)



    @staticmethod
    def get_file_info(file_path):
        # 1. 获取文件名和大小
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        # 2. 获取创建时间 (OS获取的是时间戳，转换成数据库格式)
        timestamp = os.path.getctime(file_path)
        created_date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        # 3. 计算 Hash
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()

        file_info = {'file_name': file_name,
                     'created_date': created_date,
                     'file_hash': file_hash,
                     'file_path': file_path,
                     'file_size': file_size}
        
        print(f"文件信息：{file_info}")

        return file_info

    @staticmethod
    def get_chunks_from_file(file_path, chunk_size=500, overlap=50):
        if not os.path.exists(file_path):
            return f"错误：文件路径不存在 -> {file_path}"
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # 执行切片逻辑
        chunks = []
        step = chunk_size - overlap 

        for i in range(0, len(text), step):
            start = i
            end = min(i + chunk_size, len(text))

            if len(text) > chunk_size and (start_step := end - start) != chunk_size:
                 start -= (chunk_size - start_step)

            chunk_text = text[start : end]
            chunks.append({
                'start': start,
                'end': end, 
                'text': chunk_text 
            })
        
            # 如果 end 已经等于 len(text)，说明已到尽头，循环结束
            if end == len(text):
                break

        print(f"共切分出 {len(chunks)} 个片段")
        print(f"第一个片段: {chunks[0]}")
        print(f"第二个片段: {chunks[1]}")
        print(f"末尾片段: {chunks[-1]}")    
        return chunks

    @staticmethod
    def merge_intervals(intervals):
        # 合并那些切片的start和end
        if not intervals: return []

        # 按照区间的“起始位置”对所有片段进行从小到大的排序
        intervals.sort(key=lambda x: x[0])

        merged = [intervals[0]]
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]

            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    @staticmethod
    def read_text_from_intervals(file_path, intervals):
        """
        根据文件路径和合并后的坐标集合读取文本
        :param file_path: 文件的绝对路径
        :param intervals: 合并后的坐标集合 (list of tuples)
        :return: 拼接好的文本内容
        """

        full_content = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for start, end in intervals:
                    # 移动到指定起始位置
                    f.seek(start)

                    # 读取指定的长度
                    text = f.read(end - start)
                    full_content.append(text)
        except FileNotFoundError:
            return f"错误：找不到文件 {file_path}"
        except Exception as e:
            return f"读取出错: {str(e)}"

        # 使用换行符连接不同片段，方便阅读与模型处理
        return "\n...[片段衔接]...\n".join(full_content)

    def mysql_init(self):
        # mysql 连接到数据库
        self.mysql_connection = pymysql.connect(
            host='localhost',
            user='root',
            password='tw007909',
            # database='my_db',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

        # 创建mysql 数据库和表
        with self.mysql_connection:
            with self.mysql_connection.cursor() as cursor:
                # 创建数据库（如果不存在）
                cursor.execute("CREATE DATABASE IF NOT EXISTS `my_db` DEFAULT CHARACTER SET utf8mb4;")

                # 切换到该数据库
                cursor.execute("USE `my_db`;")

                # 创建表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS `file_metadata` (
                               `id` INT AUTO_INCREMENT PRIMARY KEY,
                               `file_name` VARCHAR(255) NOT NULL,
                               `created_date` DATETIME DEFAULT CURRENT_TIMESTAMP,
                               `file_hash` VARCHAR(64) NOT NULL UNIQUE,
                               `file_path` VARCHAR(1024) NOT NULL,
                               `file_size` BIGINT DEFAULT 0
                               ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                # 连接不会自动提交。必须手动提交才能保存更改。
                self.mysql_connection.commit()

    def mysql(self):
        # mysql 连接到数据库
        self.mysql_connection = pymysql.connect(
            host='localhost',
            user='root',
            password='tw007909',
            database='my_db',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def mysql_add(self, file_name, created_date, file_hash, file_path, file_size):
        sql = " INSERT INTO `file_metadata` (`file_name`, `created_date`, `file_hash`, `file_path`, `file_size`) VALUES (%s, %s, %s, %s, %s)"
        
        data = (file_name, created_date, file_hash, file_path, file_size)

        new_id = 0
        with self.mysql_connection:
            with self.mysql_connection.cursor() as cursor:
                cursor.execute(sql, data)

                # 获取刚刚生成的自增 ID
                new_id = cursor.lastrowid
                print(f"数据已插入，生成的 ID 是: {new_id}")
            
                # 连接不会自动提交。必须手动提交才能保存更改。
                self.mysql_connection.commit()

        return new_id  # 你可以将 ID 返回，以便后续使用


    def mysql_query(self, id: int):
        """根据 ID 查询并返回 file_name, file_path, file_size"""
        sql = "SELECT `file_name`, `file_path`, `file_size` FROM `file_metadata` WHERE `id` = %s"
        
        file_info = dict()
        with self.mysql_connection:
            with self.mysql_connection.cursor() as cursor:
                cursor.execute(sql, (id, ))

                result = cursor.fetchone() 
                file_info['file_name'] = result['file_name']
                file_info['file_path'] = result['file_path']
                file_info['file_size'] = result['file_size']

        print(f"mysql查询信息: {file_info}")
        return file_info

    # 清空数据，希望保留之前定义的 Mapping（字段类型、分词器配置）
    def elasticsearch_delete(self):
        resp = self.es_client.delete_by_query(index="my_index", body={
            "query": {
                "match_all": {}
            }
        })

        print(f"已删除文档数量: {resp['deleted']}")

    def generate_actions(self, data_list):
        for item in data_list:
            yield {
                '_index': 'my_index',
                '_id': f"{item['mysql_id']}_{item['start_index']}",
                '_source': {
                    "title": item.get('title', ''),
                    "content": item.get('content', ''),
                    "mysql_id": item['mysql_id'],
                    "mysql_table": item['mysql_table'],
                    "start_index": item['start_index'],
                    "end_index": item['end_index']
                }
    }

    def elasticsearch_add(self, data_list) -> None:
        success, errors = helpers.bulk(self.es_client, self.generate_actions(data_list))

        print(f"成功导入: {success} 条")
        if errors:
            print(f"部分写入失败，失败条数: {len(errors)}")

    def elasticsearch_query(self, prompt:str, num = 3):
        resp = self.es_client.search(index="my_index", query={
            "multi_match": {
                "query": prompt,
                "fields": ["title^2", "content"],
                "minimum_should_match": "70%"
                }
            },
            size=num)

        return resp

    def get_embeddings(self, prompt:str, model_name:str = "qwen3-embedding:8b", num_ctx:int = 2048) -> list[float]:
        response = self.ollama_client.embeddings(
            model=model_name, 
            prompt=prompt,
            options={
                'num_ctx': num_ctx
            })

        vector = response['embedding']

        print(f"向量维度: {len(vector)}")
        print(f"前 5 个数字: {vector[:5]}")

        return vector
    
    def chromadb_add(self,  vector:list[float], mysql_id: int, mysql_table_name:str, start_index:int, end_index:int, content:str) -> None:
        self.chroma_collection.add(
            ids=[f"{mysql_id}_{start_index}"],
            embeddings=[vector],
            metadatas=[{"mysql_id": mysql_id, 
                    "mysql_table_name": mysql_table_name,
                    "content": content,
                    "start_index": start_index,
                    "end_index": end_index}]
        )

    def chromadb_query(self, vector:list[float], n_results:int = 3) -> dict[str, any]:
        return self.chroma_collection.query(query_embeddings=[vector], n_results=n_results)

    
    def get_query_context(self, prompt:str, c_n_results, es_n_results,  top_n = 5):
        # 结构化存储：按 mysql_id 分组
        # defaultdict(list) 会自动为每个新 ID 创建一个空列表
        grouped_intervals = defaultdict(list)


        print("=" * 50)
        results = self.chromadb_query(
            vector=self.llm_get_embedding(prompt=prompt),
            n_results= c_n_results
        )

        for res in results['metadatas'][0]:
            print(f"mysql_id: {res['mysql_id']}, start_index: {res['start_index']}, end_index: {res['end_index']}, content: {res['content']}")
            grouped_intervals[res['mysql_id']].append({
                'content_map':(res['start_index'], res['end_index']),
                'content': res['content'],
                'num': self.get_reranker(query=prompt, passage=res['content'])})

        print("=" * 50)
        resp = self.elasticsearch_query(prompt=prompt, num=es_n_results)
        print("Got %d Hits:" % resp['hits']['total']['value'])
        for hit in resp['hits']['hits']:
            source = hit["_source"]
            print(f"mysql_id: {source['mysql_id']} |  start_index: {source['start_index']} | end_index: {source['end_index']}")

            grouped_intervals[source['mysql_id']].append({
                'content_map': (source['start_index'], source['end_index']),
                'content': source['content'],
                'num': self.get_reranker(query=prompt, passage=source['content'])})

        # 建立一个“分数索引清单”
        score_index = []
        for mysql_id, intervals in grouped_intervals.items():
            for index, chunk in enumerate(intervals):
                print(f"mysql_id: {mysql_id}, content_map: {chunk['content_map']}, content: {chunk['content'][0:10]}, num: {chunk['num']}")

                score_index.append({
                    'mysql_id': mysql_id,
                    'list_index': index,     # 记录它在原列表里的位置
                    'score': chunk['num'][0] # 取出纯数值分数
                })

        # 对这个索引表进行全局排序
        score_index.sort(key=lambda x: x['score'], reverse=True)
        # 截取前 N 个最相关的“坐标”
       
        top_items = score_index[:top_n]
        print("*"*50)
        print(top_items)

        # 结构化存储：按 mysql_id 分组
        # defaultdict(list) 会自动为每个新 ID 创建一个空列表
        # 重排序后的内容
        reranker_grouped_intervals = defaultdict(list)
        for item in top_items:
            # 直接通过 mysql_id 和 list_index 反向索引回原数据
            content_map = grouped_intervals[item['mysql_id']][item['list_index']]['content_map']

            reranker_grouped_intervals[item['mysql_id']].append(content_map)

        print("*"*50)
        print(reranker_grouped_intervals)
        
        # 批量合并逻辑：遍历字典中的每一个 ID
        final_reading_tasks = {}

        for mysql_id, intervals in reranker_grouped_intervals.items():
            # 对当前 ID 下的所有区间进行合并
            final_reading_tasks[mysql_id] = self.merge_intervals(intervals)

            print(f"ID {mysql_id} 合并完成：从 {len(intervals)} 个碎片 -> {len(final_reading_tasks[mysql_id])} 个读取任务")

        print(final_reading_tasks)

        # 最终读取并整合所有上下文
        all_context = ""
        for mysql_id, intervals in final_reading_tasks.items():
            self.mysql()
            file_path = self.mysql_query(id=mysql_id)['file_path']
            content = self.read_text_from_intervals(file_path=file_path, intervals=intervals)
            all_context += f"\n=== 来自文件 {file_path} 的上下文 ===\n{content}"
            
        return all_context
        

    def file_to_mysql(self, file_path):
        # 文档信息录入mysql
        file_info = self.get_file_info(file_path)
        self.mysql()
        self.mysql_add(file_name=file_info['file_name'],
                            created_date=file_info['created_date'],
                            file_hash=file_info['file_hash'],
                            file_path=file_info['file_path'],
                            file_size=file_info['file_size'])

    def file_to_es(self, mysql_id, mysql_table, file_path):
        # 文档内容切片录入Elasticsearch
        es_chunks = self.get_chunks_from_file(file_path=file_path,
                                                chunk_size=1000,
                                                overlap= 150)
        
        # 快速转换
        data_list = [{
            'title': item['text'][0:20].strip(),
            'content': item['text'],
            'mysql_id': mysql_id,
            'mysql_table': mysql_table,
            'start_index': item['start'],
            'end_index': item['end']
        } for item in es_chunks]

        print(f"data_list数据: {data_list[-1]}")

        self.elasticsearch_add(data_list)

    def file_to_ChromaDB(self, mysql_id, mysql_table, file_path):
        # 文档内容切片录入ChromaDB
        cdb_chunks = self.get_chunks_from_file(file_path=file_path,
                                                chunk_size=400,
                                                overlap= 60)
        for chunks in cdb_chunks:
            self.chromadb_add(
                vector=self.get_embeddings(prompt=chunks['text']),
                mysql_id=mysql_id,
                mysql_table_name=mysql_table,
                start_index=chunks['start'],
                end_index=chunks['end'],
                content=chunks['text'])

    def chat(self, prompt, num_ctx_k=32, top_k=100, min_p=0.1, top_p=0.9):
        response = self.ollama_client.chat(model='huihui_ai/Qwen3.6-abliterated:35b', messages=[
        {
            'role': 'user',
            'content': f"{prompt}",
            # 'images': [f"{img}"]
        }],
        options={
            'num_ctx': num_ctx_k * 1024,
            "top_k": top_k,
            "min_p": min_p,
            "top_p": top_p
            })

        print("ollama输出:")
        print(response.message.content)
        return response.message.content
    
    def get_reranker(self, query:str, passage:str):
        # 对文本进行打分
        return self.reranker.compute_score([query, passage], normalize=True)

    def agent(self, user_question=None, log=None):
        
        context_data = self.get_query_context(prompt=user_question, c_n_results=20, es_n_results=10, top_n=5)

        search_num = 10
        final_answer = r""
        archive_content = r""

        agent_num = 0
        while True:

            agent_num += 1
            
            print("*"*30, f"第{agent_num}轮","*"*30)

            log.write(f"{'*'*30} 第{agent_num}轮 {'*'*30} \n") if log else None
            # 强制存盘
            log.flush()

            if search_num <= -20:
                break

            # 构造最终的 Prompt
            final_prompt = f"""  
            ### 上轮总结
            {archive_content}

            ### 上轮回答: 
            {final_answer}

            ### 参考资料：
            {context_data}

            ### 用户问题：
            {user_question}
            """


            print("="*60)
            print(final_prompt)
            print("="*60)

            raw_json_response, tokens_use = self.llm_chat(content=final_prompt, search_num=20)

            log.write(f"{raw_json_response} \n") if log else None
            log.write(f"tokens消耗:{tokens_use} \n") if log else None

            
            try:
                data = json.loads(raw_json_response)

                if data["action"] == "SEARCH" and search_num > 0:
                           

                    if data['archive_update']['action'] == "APPEND":
                        archive_content += f"第{agent_num}轮总结:\n{data['archive_update']['content']}\n"
                    elif data['archive_update']['action'] == "REPLACE":
                        archive_content = data['archive_update']['content']
                    else:
                        pass

                    print("*"*50)
                    print(f"总结信息: {archive_content}")
                    log.write(f"{'*'*50} \n 总结信息: {archive_content} \n") if log else None


                    final_answer = data['final_answer']
                    print("*"*50)
                    print(f"ollama回答: {final_answer}")
                    log.write(f"{'*'*50} \n {final_answer} \n") if log else None

                    print("*"*50)
                    print(f"获取查询词为: {data['search_query']}")

                    log.write(f"{'*'*50} \n {data['search_query']} \n") if log else None                   

                    new_user_question = ",".join(data['search_query'])
                    log.write(f"{'*'*50} \n 新查询词： {new_user_question} \n") if log else None
                    
                    context_data = self.get_query_context(prompt=new_user_question, c_n_results=20, es_n_results=10, top_n=5)
                    search_num -= 1
                else:
                    print(f"最终答案: {data['final_answer']}")
                    log.write(f"{'*'*50} \n 最终答案: {data['final_answer']} \n") if log else None
                    break
            except json.JSONDecodeError:
                print("模型未返回标准 JSON，请检查模型状态")
                break
        
    def llm_chat(self, search_num=10, content=""):

        user_json = {
            "search_num": {
                "type": "integer",
                "enum": [search_num],  # 锁死数字，模型只能填这个外部传入的值
                "description": "当前剩余的查询额度"
            },
            "action": {
                "type": "string",
                "enum": ["SEARCH", "FINAL_ANSWER"],  # 限定只能在这两个词里选
                "description": "有余额，允许根据需求选择搜索或直接回答"
            },
            "search_query": {
                "type": "array",
                "description": "从不同侧面、多维度拆分出的精准搜索关键词列表。",
                "items": {
                    "type": "string",
                    "minLength": 2,      # 每个关键词最少2个字，防止吐出废词
                    "maxLength": 30      # 限制单组词长度
                },
                "minItems": 10, # 硬性规定大模型必须吐出【至少10组】不同的检索词！
                "maxItems": 20           # 规定上限，防止模型无脑长篇大论
            },
            "final_answer": {
                "type": "string",
                "description": "当前回复[用户问题]的回答"
            },
            "archive_update": {
                "type": "object", 
                "description": "历史记录 history.md 的更新策略",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["APPEND", "REPLACE", "NONE"],  # 限定可选值
                        "description": "文档更新动作"
                    },
                    "content": {
                        "type": "string",
                        "description": "此处填写需要追加到 history.md 的核心信息（如：结论、新任务、重要设定等等）"
                    }
                },
                "required": ["action", "content"]  # 内部对象必填字段
            }
        }

        output = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个能以 JSON 格式输出数据的实用中文AI助手。请根据[用户问题]中的要求，参考[上轮回答]和[参考资料]和[上轮总结]进行回答",
                },
                {"role": "user", "content": content},
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": user_json,
                    "required": ["action", "search_query", "final_answer", "archive_update"],
                },
            },
            temperature=0.7,
        )

        print("*"*60)
        print(output)
        print(output['choices'][0]['message']['content'])
        print(output['usage'])

        return output['choices'][0]['message']['content'], output['usage']

    def llm_get_embedding(self, prompt:str):
        embeddings = self.llm_embedding.create_embedding(prompt)

        print("*"*60)
        #print(embeddings)
        # print(embeddings['data'][0]['embedding'])
        print(embeddings['usage'])

        return embeddings['data'][0]['embedding']

    def llm_close(self):
        self.llm.close()
        self.llm_embedding.close()



if __name__ == "__main__":

    FU01_ollama = FU01()

    file_path = r"G:\Doc\《一念永恒》（校对版全本）作者：耳根 @www.zei8.me.txt"
    mysql_id = 6
    mysql_table = 'file_metadata'


    

    
    user_question = r"给我介绍一下佳树"

    with open(r"G:\img_WD14\log\log.txt", "a", encoding="utf-8") as log:
        FU01_ollama.agent(user_question=user_question, log=log)

        # 强制存盘
        log.flush()
    
    FU01_ollama.llm_close()