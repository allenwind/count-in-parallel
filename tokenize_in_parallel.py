from multiprocessing import Pool, Queue
from count_in_parallel import *

def batch_generator(generator, batch_size=300):
    batch_texts = []
    for i, text in enumerate(generator):
        batch_texts.append((i, text))
        if len(batch_texts) == batch_size:
            yield batch_texts
            batch_texts = []
    if batch_texts:
        yield batch_texts

@timethis
def tokenize_in_parallel(
    tokenize,
    generator,
    processes=7,
    maxsize=300,
    preprocess=None):
    if preprocess is None:
        preprocess = lambda x: x
    
    def batch_tokenize(batch_texts_queue, tokens_queue):
        while True:
            batch_tokens = []
            batch_texts = batch_texts_queue.get()
            for i, text in batch_texts:
                text = preprocess(text)
                tokens = tokenize(text)
                batch_tokens.append((i, tokens))
            tokens_queue.put(batch_tokens)
    
    batch_texts_queue = Queue(maxsize)
    tokens_queue = Queue()
    pool = Pool(processes, batch_tokenize, initargs=(batch_texts_queue, tokens_queue))

    gtokens = []
    def merge():
        batch_tokens_size = 0
        for _ in range(tokens_queue.qsize()):
            batch_tokens = tokens_queue.get()
            batch_tokens_size += 1
            for i, tokens in batch_tokens:
                gtokens.append((i, tokens))
        return batch_tokens_size

    batch_tokens_size = 0
    for batch_texts_size, batch_texts in enumerate(batch_generator(generator), start=1):
        while True:
            try:
                batch_texts_queue.put(batch_texts, block=False)
                break
            except queue.Full:
                batch_tokens_size += merge()

            if batch_texts_size % (processes * maxsize // 2) == 0:
                batch_tokens_size += merge()

    while batch_tokens_size != batch_texts_size:
        batch_tokens_size += merge()

    pool.terminate()
    gtokens.sort(key=lambda x: x[0])
    gtokens = [i[1] for i in gtokens]
    return gtokens