import itertools
import jieba
from count_in_parallel import *

file = "/home/zhiwen/workspace/dataset/THUCNews-title-label.txt"

def gen(file):
    with open(file, encoding="utf-8") as fp:
        text = fp.read()
    lines = text.split("\n")[:-1]
    for line in lines:
        yield line

hybrid_tokenize = lambda x: itertools.chain((i for i in x), jieba.cut(x))

tokens = count_in_parallel_from_generator(
    tokenize=hybrid_tokenize,
    generator=gen(file),
    processes=6,
    maxsize=300,
    preprocess=lambda x: x.lower()
)
print(len(tokens))
print(tokens.most_common(200))
print(tokens.most_common(-200))
