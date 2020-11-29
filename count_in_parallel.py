import time
import random
import glob
import queue
import collections
import heapq
from functools import wraps
from operator import itemgetter
from multiprocessing import Pool, Queue

basic_tokenize = lambda text: list(text)

def load_batch_texts(path, batch_size=300, limit=None, shuffle=True):
    # 批量的形式加载文本
    files = glob.glob(path)[:limit]
    if not files:
        raise FileNotFoundError("without any files")
    if shuffle:
        random.shuffle(files)
    batch_texts = []
    for file in files:
        with open(file, "r", encoding="utf-8") as fd:
            text = fd.read()
        batch_texts.append(text)
        if len(batch_texts) == batch_size:
            yield batch_texts
            batch_texts = []
    if batch_texts:
        yield batch_texts

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, "{:.3f}s".format(end-start))
        return result
    return wrapper

class Counter(collections.Counter):

    def most_common(self, n=None):
        # 支持传入负数，表示least common
        if n is None:
            return sorted(self.items(), key=itemgetter(1), reverse=True)
        if n > 0:
            return heapq.nlargest(n, self.items(), key=itemgetter(1))
        return self.least_common(-n)

    def least_common(self, n):
        # 最不常见
        return heapq.nsmallest(n, self.items(), key=itemgetter(1))

@timethis
def count_in_parrallel(tokenize, batch_generator, processes, maxsize=300):
    def batch_counter(batch_texts_queue, tokens_queue):
        # 批量统计
        while True:
            tokens = Counter()
            batch_texts = batch_texts_queue.get()
            for text in batch_texts:
                for token in tokenize(text):
                    tokens[token] += 1
            tokens_queue.put(tokens)

    # 数据队列
    batch_texts_queue = Queue(maxsize)
    tokens_queue = Queue()
    
    # 进程池
    pool = Pool(processes, batch_counter, initargs=(batch_texts_queue, tokens_queue))

    # 全局统计表
    gtokens = Counter()
    def merge_tokens():
        # 合并每个进程的统计表
        batch_tokens_size = 0
        for _ in range(tokens_queue.qsize()):
            tokens = tokens_queue.get()
            batch_tokens_size += 1
            for k, v in tokens.items():
                gtokens[k] += v
        return batch_tokens_size

    batch_tokens_size = 0
    for batch_texts_size, batch_texts in enumerate(batch_generator, start=1):
        while True:
            try:
                batch_texts_queue.put(batch_texts, block=False)
                break
            except queue.Full:
                batch_tokens_size += merge_tokens()

    while batch_tokens_size != batch_texts_size:
        batch_tokens_size += merge_tokens()

    pool.terminate()
    return gtokens

if __name__ == "__main__":
    # 测试
    import jieba
    path = "/home/zhiwen/workspace/dataset/THUCTC/THUCNews/**/*.txt"
    tokens = count_in_parrallel(
        tokenize=jieba.lcut,
        batch_generator=load_batch_texts(path, limit=10000),
        processes=6,
        maxsize=300
    )
    print(len(tokens))
    print(tokens.most_common(200))
    print(tokens.most_common(-200))
