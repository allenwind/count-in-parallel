import jieba
from count_in_parallel import *

path = "/home/zhiwen/workspace/dataset/THUCTC/THUCNews/**/*.txt"
files = glob.glob(path)

tokens = count_in_parallel_from_files(
    tokenize=jieba.lcut,
    files=files,
    processes=6,
    limit=10000
)
print(len(tokens))
print(tokens.most_common(200))
print(tokens.most_common(-200))
