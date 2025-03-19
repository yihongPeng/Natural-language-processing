import random
import torch
from text_preprocessing import load_corpus_text
from text_preprocessing import Vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0,num_steps - 1):]
    num_subseqs = (len(corpus)-1) // num_steps
    initial_indices = list(range(0,num_subseqs * num_steps,num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos:pos + num_steps]
    num_batches = num_subseqs // batch_size
    for i in range(0,batch_size * num_batches,batch_size):
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        x = [data(j) for j in initial_indices_per_batch]
        y = [data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(x),torch.tensor(y)

def seq_data_iter_sequential(corpus,batch_szie,num_steps):
    offset = random.randin(0,num_steps)
    return

if __name__ == '__main__':
    corpus , vocab = load_corpus_text()
    # corpus = [vocab.idx2token[idx] for idx in corpus]

    # 二元语法
    bigram_tokens = [pair for pair in zip(corpus[:-1],corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens) 
    print(bigram_vocab.token_freqs[:5])
    for x,y in seq_data_iter_random(corpus,batch_size=2,num_steps=5):
        print(f"x: {x},y: {y}")
    
