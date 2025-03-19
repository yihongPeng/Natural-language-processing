import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from text_preprocessing import load_corpus_text
from LanguageModel import seq_data_iter_random
from text_preprocessing import Vocab
import tqdm
from tqdm import tqdm
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device) * 0.01
    
    W_xh = normal((num_inputs,num_hiddens))
    W_hh = normal((num_hiddens,num_hiddens))
    b_h = torch.zeros(num_hiddens,device=device)
    W_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeros(num_outputs,device=device)
    params = [W_xh, W_hh, b_h, W_hq,b_q]
    for param in params:
        param.requires_grad = True
    return params

# 初始化隐藏状态
def init_rnn_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)

def rnn(inputs, state, params):
    W_xh,W_hh,b_h,W_hq,b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) 
                       + torch.mm(H, W_hh)
                       + b_h)
        Y = torch.mm(H,W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs,dim = 0), (H,)

class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.vocab_size,self.num_hiddens = vocab_size,num_hiddens
        self.params = get_params(vocab_size,num_hiddens,device)
        self.init_state,self.forward_fn = init_state,forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T,self.vocab_size).type(torch.float32)
        return self.forward_fn(X,state,self.params)
    
    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,
                               self.num_hiddens,
                               device)

def predict(prefix:str,num_preds:int,net,vocab:Vocab,device:str):
    # 在prefix后面生成新字符
    state = net.begin_state(batch_size=1,device=device)
    outputs = []
    get_input = lambda: torch.tensor(outputs[-1],device=device).reshape(1,1)
    for token in prefix:
        _,state = net(torch.tensor(vocab[token]).reshape(1,1),state)
        outputs.append(vocab[token])
    for i in range(num_preds):
        y,state = net(get_input(),state)
        outputs.append(int(y.argmax(dim=1)))
    return ' '.join([vocab.idx2token[idx] for idx in outputs])

def grad_clipping(net:RNNModelScratch,theta):
    if isinstance(net,nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum()))

def train_iter(net: RNNModelScratch,corpus: list,num_epoches = 10,batch_size = 16,num_steps = 64):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.params,lr=0.001)

    for epoch in range(num_epoches):
        total_loss = 0
        state = net.begin_state(batch_size,device='cpu')
        pbar = tqdm(seq_data_iter_random(corpus,batch_size,num_steps))
        i = 0
        for X,y in pbar:
            i += 1
            # pbar.set_description(desc=f'epoch: {i}')
            y = y.reshape(-1)
            state[0].detach_()
            y_target,state = net(X,state)
            
            optimizer.zero_grad()
            l = loss(y_target,y.long())
            l.backward()
            pbar.set_postfix(loss=l)
            pbar.update(1)
            optimizer.step()
            total_loss += l.item()
        pbar.write(s=f'mean loss:{total_loss/i}')
        
if __name__ == '__main__':
    batch_size, num_steps ,num_epoches= 32, 128, 20
    corpus, vocab = load_corpus_text(mode = 'char')
    num_hiddens = 512
    net = RNNModelScratch(len(vocab),num_hiddens,'cpu',get_params,init_rnn_state,rnn)
    # for x,y in seq_data_iter_random(corpus,batch_size,num_steps):
    #     state = net.begin_state(batch_size,'cpu')
    #     Y,new_state = net(x,state)
    #     print(x.shape,Y.shape,state[0].shape,new_state[0].shape)
    #     break
    print(net(torch.ones((1,1),dtype=torch.long),net.begin_state(1,device='cpu'))[0].argmax(dim=1).reshape(1).shape)
    print(predict(prefix=['i'],num_preds=20,net=net,vocab=vocab,device='cpu'))
    train_iter(net,corpus,num_epoches,batch_size,num_steps)
