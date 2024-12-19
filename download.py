from modelscope import HubApi
from modelscope import snapshot_download

api=HubApi()
api.login('2ac8c899-2f0a-4397-ad37-661532c85710')

# download your model, the model_path is downloaded model path.
model_path =snapshot_download(model_id='BAAI/bge-m3',cache_dir='./your_project')
model_path =snapshot_download(model_id='BAAI/bge-reranker-v2-m3',cache_dir='./your_project')
model_path =snapshot_download(model_id='sentence-transformers/paraphrase-MiniLM-L6-v2',cache_dir='./your_project')