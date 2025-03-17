import collections
import re

# 简单预处理:转小写,出去非字母元素

def read_text(fn = 'wizard_of_os.txt'):
    with open(fn,'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines]

def  tokenize(lines,token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('Error: Unknown token type' + token)

# 计算每个词出现的频率(次数)

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0],list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    # min_freq,出现次数小于min_freq不计入词汇表
    def __init__(self,tokens=None,min_freq = 0,reserved_tokens=None):
        if tokens is None:
            tokens =[]
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(),key=lambda x:x[1],reverse=True)
        self.idx2token = ['<unk>'] + reserved_tokens
        self.token2idx = {
            token: idx for idx,token in enumerate(self.idx2token) 
        }
        for token,freq in self.token_freqs:
            if freq <= min_freq:
                break
            if token not in self.token2idx:
                self.idx2token.append(token)
                self.token2idx[token] = len(self.idx2token) - 1
    def __len__(self):
        return len(self.idx2token)
    def __getitem__(self,tokens):
        return [self.token2idx[token] for token in tokens]

if __name__ == '__main__':
    lines = read_text()
    tokens = tokenize(lines,'word')
    vocab = Vocab(tokens)
    print(vocab[['i', 'am','the', 'one']])
