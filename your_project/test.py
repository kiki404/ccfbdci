#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
# 创建参数解析器
parser = argparse.ArgumentParser(description="Test script to accept dataset path")
parser.add_argument('dataset_dir', type=str, help='Path to the dataset directory')

# 解析传入的命令行参数
args = parser.parse_args()
test_path = args.dataset_dir
test_path = test_path.strip()
test_path = os.path.join(test_path, "test2.jsonl")
# 现在可以通过 args.dataset_dir 访问传递的路径参数
print(f"Dataset directory is: {args.dataset_dir}")


# In[ ]:


from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS, CHAT_MODELS
from typing import Dict
MY_MODELS: Dict[str, int] = {
    "qwen-plus": 32768,
    "gpt-3.5-turbo": 4000,
    "moonshot-v1-8k": 8000,
    "llama3-70b-8192": 8192,
}
ALL_AVAILABLE_MODELS.update(MY_MODELS)
# 不加入这个字典，会导致它采用Completion而不是Chat Completion接口，Qwen不兼容Completion兼容。
CHAT_MODELS.update(MY_MODELS)


# In[ ]:


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings,StorageContext,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
import torch
from llama_index.llms.openai import OpenAI

# selected_model = "/mnt/sda/cl/competition/BDCI/Qwen2.5-7B-Instruct"


llm =  OpenAI(
    model="qwen-plus",
    api_key="sk-7ad0f57d5ec54e888b5d308d084afe1e", 
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature = 0.2

)
embed_model = HuggingFaceEmbedding(model_name="./your_project/BAAI/bge-m3")

Settings.chunk_size = 512                        
Settings.embed_model = embed_model
# Settings.embed_model = HuggingFaceEmbedding(model_name="/mnt/sda/cl/competition/BDCI/bge-large-zh-v1.5")
Settings.llm = llm


# In[ ]:


from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("./your_project/corpus/userdoc",recursive=True,filename_as_id=True).load_data()


# In[ ]:


from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter,SemanticSplitterNodeParser
base_splitter  = SentenceSplitter(chunk_size=512,chunk_overlap=128)

nodes = base_splitter.get_nodes_from_documents(documents)
index = VectorStoreIndex(
    nodes=nodes,
    show_progress=True,
)


import json
import logging
import os

from typing import Any, Callable, Dict, List, Optional, cast
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)

import bm25s
import itertools
import jieba
from nltk.corpus import stopwords
stops=set(stopwords.words('chinese'))
class ChineseBM25Retriever(BaseRetriever):
    """A BM25 retriever that uses the BM25 algorithm to retrieve nodes.

    Args:
        nodes (List[BaseNode], optional):
            The nodes to index. If not provided, an existing BM25 object must be passed.
        similarity_top_k (int, optional):
            The number of results to return. Defaults to DEFAULT_SIMILARITY_TOP_K.
        callback_manager (CallbackManager, optional):
            The callback manager to use. Defaults to None.
        objects (List[IndexNode], optional):
            The objects to retrieve. Defaults to None.
        object_map (dict, optional):
            A map of object IDs to nodes. Defaults to None.
        verbose (bool, optional):
            Whether to show progress. Defaults to False.
    """


    def _chinese_tokenizer(self, texts: List[str]) -> tuple[str]:
        # Use jieba to segment Chinese text
        rslts= tuple(itertools.chain.from_iterable(jieba.cut(text) for text in texts))
        return rslts
    
    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,               
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:
        
        self.similarity_top_k = similarity_top_k


        self.stop_words = set(stopwords.words('chinese'))

        corpus_tokens = [
            [word for word in jieba.cut_for_search(node.get_content()) if word not in self.stop_words]                                  
            for node in nodes
        ]
        self.bm25 = bm25s.BM25()
        from llama_index.core.vector_stores.utils import (
            node_to_metadata_dict
        )
        
        corpus = [node_to_metadata_dict(node) for node in nodes]
        self.bm25.corpus = corpus
        self.bm25.index(corpus_tokens, show_progress=True)

        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str

        tokenized_query = [[word for word in jieba.cut_for_search(query) if word not in self.stop_words]]
        
        indexes, scores = self.bm25.retrieve(
            tokenized_query, k=self.similarity_top_k, show_progress=self._verbose
        )

        # batched, but only one query
        indexes = indexes[0]
        scores = scores[0]

        nodes: List[NodeWithScore] = []
        for idx, score in zip(indexes, scores):
            # idx can be an int or a dict of the node
            if isinstance(idx, dict):
                node = metadata_dict_to_node(idx)
            else:
                node_dict = self.corpus[int(idx)]
                node = metadata_dict_to_node(node_dict)
            nodes.append(NodeWithScore(node=node, score=float(score)))

        return nodes



query_gen_prompt_str = (
    "将下面的句子翻译成英文,直接输出翻译后的结果"
    "原始句子：{query}"
    "翻译后的句子："
)
query_gen_prompt = PromptTemplate(query_gen_prompt_str)
def translate_en(llm,query_str:str):
    fmt_prompt = query_gen_prompt.format(
        query = query_str,
    )
    response = llm.complete(fmt_prompt).text
    return response
query = "存储Point类型数据的格式是什么？"
rewrite_query = translate_en(llm,query)
print(rewrite_query)


# In[ ]:


import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-xxx", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
query_gen_prompt_str = (
    "请你对以下用户提出的问题进行改写，遵循以下修改规则：\n"
    "1. 改写后的问题需保留原句关键信息和核心意图，避免冗余改动。\n"
    "2. 对于意图不明确的问题，请添加一种适当的疑问词，使问题更清晰。\n"
    "3. 若句子结构已经简洁明了且主旨明确，则采用最小改动原则。\n"
    "4. 对于明显存在两个或以上子问题的句子，将其拆分为两个独立的完整问题。\n"
    "5. 句子中的英文词汇不要变动\n"
    "6. 直接输出改写后的问题，无需解释。\n"
    "原始问题：{query}\n"
    "改写后的问题："
)
def query_rewrite(llm,query_str:str):
    fmt_prompt = query_gen_prompt_str.format(
            query = query
        )
    completion = client.chat.completions.create(
        model="qwen2.5-72b-instruct", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': fmt_prompt}],
        temperature = 0.2
        )
    response = completion.choices[0].message.content
    return response
query = "tugraph可以最多创建多少点边和点边上最多创建多少属性？"
response = query_rewrite(llm,query)
print(response)




from llama_index.core.retrievers import VectorIndexRetriever
import copy
class EnglishVectorRetriever(VectorIndexRetriever):
    def _retrieve(self,query_bundle: QueryBundle,) -> List[NodeWithScore]:
        query_bundle_en = copy.copy(query_bundle)
        query_str = query_bundle_en.query_str
        English_query_str = translate_en(llm,query_str)
        query_bundle_en.query_str = English_query_str
        if self._vector_store.is_embedding_query:
            query_bundle_en.embedding = None
            if query_bundle_en.embedding is None and len(query_bundle_en.embedding_strs) > 0:
                query_bundle_en.embedding = (
                    self._embed_model.get_agg_embedding_from_queries(
                        query_bundle_en.embedding_strs
                    )
                )

        return self._get_nodes_with_embeddings(query_bundle_en)




from llama_index.core.retrievers import VectorIndexRetriever
import re
import copy
class MyVectorRetriever(VectorIndexRetriever):
    def _retrieve(self,query_bundle: QueryBundle,) -> List[NodeWithScore]:
        query_bundle_remove_tugraph_str = re.sub(r'(?i)tugraph', "", query_bundle.query_str).strip()
        query_bundle_remove_tugraph = copy.copy(query_bundle)
        query_bundle_remove_tugraph.query_str = query_bundle_remove_tugraph_str

        if self._vector_store.is_embedding_query:
            query_bundle_remove_tugraph.embedding = None
            if query_bundle_remove_tugraph.embedding is None and len(query_bundle_remove_tugraph.embedding_strs) > 0:
                query_bundle_remove_tugraph.embedding = (
                    self._embed_model.get_agg_embedding_from_queries(
                        query_bundle_remove_tugraph.embedding_strs
                    )
                )

        return self._get_nodes_with_embeddings(query_bundle_remove_tugraph)



from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_source_node

from llama_index.core import Document



query = "TuGraph 中使用的两种主要图分析操作是什么？"
## 向量检索
vector_retriever = MyVectorRetriever(
    index,
    similarity_top_k=30
)
 
## bm25 关键词检索
bm25_retriever = BM25Retriever.from_defaults(
    # docstore=index.docstore,
    nodes = nodes,
    similarity_top_k=3,
    # tokenizer=chinese_tokenizer
)
ChineseBM25_Retriever = ChineseBM25Retriever(nodes=nodes,similarity_top_k=15)
EnglishVector_Retriever = EnglishVectorRetriever(index,similarity_top_k=15)


# Nodes = EnglishVector_Retriever.retrieve(query)
# Nodes = ChineseBM25_Retriever.retrieve(query)
Nodes = vector_retriever.retrieve(query)
for node in Nodes:
    print(node)
    # display_source_node(node)
# print(vector_retriever.retrieve(query))
# print(bm25_retriever.retrieve(query))


# In[ ]:


from llama_index.core.retrievers import QueryFusionRetriever

# 定义混合retreiver
retriever = QueryFusionRetriever(
    [vector_retriever, ChineseBM25_Retriever,EnglishVector_Retriever],
    retriever_weights=[0.6, 0.3,0.1],
    similarity_top_k=40,
    num_queries=1,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=False,
    verbose=True,
)

nodes_with_scores = retriever.retrieve(
    "应该如何写入图数据库中的顶点数据？"
)
# print(nodes_with_scores)
for node in nodes_with_scores:
    # print(f"Score: {node.score:.2f} - {node.text}...\n-----")
    print(node.get_content()+'------------')



import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
val = pd.read_csv("./your_project/vals.csv")
# val = val[10:].reset_index(drop=True)
# val = df.sample(n=40)
# val_drop = df.drop(val.index).reset_index(drop=True)
# val = val.reset_index(drop=True)
model = SentenceTransformer("./your_project/sentence-transformers/paraphrase-MiniLM-L6-v2")
embeddings = model.encode(val['input_field'])
def getshots(query):
    # val = pd.read_csv("./your_project/vals.csv")
    # val = val.drop(index=i)
    # val = val.reset_index(drop=True)
    embeddings = model.encode(val['input_field'])
    query_embed = model.encode(query)
    
    query_embedding = query_embed.reshape(1, -1)
    
    # 计算余弦相似度
    similarities = cosine_similarity(embeddings, query_embedding).flatten()
    
    # 按相似度从高到低排序，提取前 4 个索引
    top_5_indices = np.argsort(similarities)[-38:][::-1]
    # 获取最相似的5个句子 embedding
    top_5_embeddings = embeddings[top_5_indices]
    # top_5_sentences = val.loc[16]["input_field"]
    example_shots = ""
    for index,i in enumerate(top_5_indices):
        Q = val.loc[i]["input_field"]
        A = val.loc[i]["output_field"]
        S = val.loc[i]["style"]
        example_shots += "{index}.问答对示例：\n".format(index = index+1)   + "Q:" + Q + "\n"  + "A:" + A + "\n"   + S + "\n\n"
    qa_prompt_tmpl_str = (
        "We have provided context information below. \n"
        # "如果上下文给出了非常明确的答案，那么请优先以上下文的原文为主\n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
"""请参考以下问答对，并注意学习这些回答的风格特点：\n
{example_shots}请根据这些示例，按照如下风格回答接下来的问题：
    - 回答应简洁直接，不包含多余的描述。
    - 无需重复问题，直接给出答案。
    - 保持回答正式且专业。
    - 回答中如果出现Tugraph为主语将其省略。
    - 回答要避免举例说明。
"""
        "Given this information, please answer the question: {query_str}\n"
    )
    return qa_prompt_tmpl_str.format(context_str="{context_str}",
        example_shots=example_shots,
        query_str="{query_str}")
print(getshots("RPC 及 HA 服务中，verbose 参数的设置有几个级别？"))


# In[ ]:


from llama_index.core import QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.indices.utils import default_parse_choice_select_answer_fn
import re

def custom_parse_choice_select_answer_fn(answer: str, num_choices: int):
    matches = re.findall(r"(Doc: \d+, Relevance: \d+)", answer)
    # 按行输出
    answer = ""
    for match in matches:
        answer += match+"\n"
    _answer = answer
    return default_parse_choice_select_answer_fn(_answer, num_choices)

rerank = SentenceTransformerRerank(model="./your_project/BAAI/bge-reranker-v2-m3", top_n =5 )
# rerank = SentenceTransformerRerank(model="/mnt/sda/cl/competition/BDCI/bge-reranker-large", top_n =5 )
# 定义query engine
qa_prompt_tmpl_str = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    """请参考以下问答对，并注意总结这些回答的风格特点：\n
    1. 问答对示例：\n
    Q: RPC 及 HA 服务中，verbose 参数的设置有几个级别？\n
    A: 三个级别（0，1，2）。\n
    （回答风格：简洁明了，直接给出结果，没有多余描述。）\n
    
    2. 问答对示例：\n
    Q: 如果成功修改一个用户的描述，应返回什么状态码？\n
    A: 200\n
    （回答风格：技术精准，只给出具体的数值，没有解释。）\n
    
    3. 问答对示例：\n
    Q: TuGraph-DB的日志等级如何调整？\n
    A: 单机模式下，调整配置文件 src/server/lgraph_standalone.json，其中 verbose 配置项控制日志等级，verbose 可以设置为 0, 1, 2，对应日志等级可以参考 src/server/lgraph_server.cpp 中 115 行至 128 行。
    （回答风格：结构化且明确，提供了操作步骤和具体路径，信息详细但不冗余。）\n
    
    4. 问答对示例：\n
    Q: 如果要在 FrontierTraversal 中并行执行遍历，事务的哪种模式必须被选用？\n
    A: 事务必须是只读的。\n
    （回答风格：直接回答，精确描述，没有额外信息。）\n
    
    5. 问答对示例：\n
    Q: 在线全量导入 TuGraph 时，如果发生数据包错误，默认行为是什么？\n
    A: 默认行为是在第一个错误包处停止导入。\n
    （回答风格：简单直接，清晰描述默认行为。）\n
    
    请根据这些示例，按照如下风格回答接下来的问题：\n
    - 回答应简洁直接，不包含多余的描述。
    - 技术术语应准确无误，避免使用模糊的词语。
    - 对于涉及参数、配置的回答，提供明确的数值和路径信息。
    - 无需重复问题，直接给出答案。
    - 保持回答正式且专业。 """
    "Given this information, please answer the question: {query_str}\n"
    
    # "回答要直接简洁，且避免重复回答，回答限制在一两句话之内!"
    # "回答直接给出结论,以上下文的原文为主。\n"
    # "若给出的上下文不足以得出正确答案，请给予恰当的回复，如暂不支持等。\n"
)

query = "调用算法 `algo.shortestPath` 的实际应用场景是什么？"
qa_prompt_tmpl_str = getshots(query)
# print(qa_prompt_tmpl_str)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine = RetrieverQueryEngine.from_args(
    retriever,
    text_qa_template = qa_prompt_tmpl,
    node_postprocessors=[rerank],
)
# rewrite_query = query_rewrite(llm,query)
print(query)
response = query_engine.query(query)
print(response.response+"-------------答案结束")
query_bundle = QueryBundle(query)
nodes = rerank.postprocess_nodes(retriever.retrieve(query),query_bundle)
for node in nodes:
    print(node.get_content()+"------------")


# In[ ]:


import pandas as pd
import json
from tqdm import tqdm
import re
# 读取 JSON 文件为 DataFrame
file_path = test_path
# file_path = '/mnt/sda/cl/competition/BDCI/val.json'
# file_path = '/mnt/sda/cl/competition/BDCI/test1.json'
# rerank = LLMRerank(llm=llm,top_n=3,parse_choice_select_answer_fn=custom_parse_choice_select_answer_fn)
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# 转换为 DataFrame
df = pd.DataFrame(data)
query_responses = []
# 遍历每个 input_field
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    query = row['input_field']
    qa_prompt_tmpl_str = getshots(query)
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        text_qa_template = qa_prompt_tmpl,
        node_postprocessors=[rerank],
    )
    rewrite_query = query_rewrite(llm,query)
    # print("问题: "+query+"\n")
    print("改写后:"+rewrite_query+"\n")
    response = query_engine.query(rewrite_query)  # 调用查询引擎
    print(response.response)
    query_responses.append(response.response)  # 将响应结果添加到列表中
# 将查询结果列表添加到 DataFrame 的新列 'query_response'

# test
df['output_field'] = query_responses
# val
# df['query_response'] = query_responses

# 显示结果
df.head(5)


# In[ ]:


import os
# 1. 选择需要导出的列
selected_df = df[['id', 'output_field']]
output_dir = "../output"
output_file = os.path.join(output_dir, "answer.jsonl")
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 避免重复创建时报错
    print(f"Directory {output_dir} created.")
# 2. 将 DataFrame 导出为 JSONL 格式的文件
selected_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
print("结果已导出到："+output_file)

output_dir = "/output"
output_file = os.path.join(output_dir, "answer.jsonl")
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 避免重复创建时报错
    print(f"Directory {output_dir} created.")
# 2. 将 DataFrame 导出为 JSONL 格式的文件
selected_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
print("结果已导出到："+output_file)

