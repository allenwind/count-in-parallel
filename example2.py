import jieba
from count_in_parallel import *

path = "/home/zhiwen/workspace/dataset/THUCTC/THUCNews/**/*.txt"
files = glob.glob(path)

tokens = count_in_parallel(
    tokenize=jieba.lcut,
    batch_generator=load_batch_texts(files, limit=10000),
    processes=6,
    maxsize=300,
    preprocess=lambda x: x.lower()
)
print(len(tokens))
print(tokens.most_common(200))
print(tokens.most_common(-200))
