import torch
from torch import nn

import jieba
from torchtext import vocab


class NewsNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        super(NewsNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)              
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embeds = self.embedding(x)
        r_out, (h_n, h_c) = self.lstm(embeds, None)  # 全0初始化h0
        # r_out : [batch, time_step, hidden_size]
        # h_n: [n_layers, batch, hidden_size]
        # h_c: [n_layers, batch, hidden_size]
        out = self.fc1(r_out[:, -1, :])   # 选取最后一个时间点的out
        return out      

    def predict(self, x, word2index, labelMap):
        X = KanjiSentence2tensor(x, word2index)
        out = self(X).argmax(1)[0].item()
        return labelMap[out]

def KanjiSentence2tensor(sentence, word2index):
    index_seq = []
    for w in jieba.lcut(sentence):
        if w not in word2index:
            index_seq.append(word2index['<unk>'])
        else:
            index_seq.append(word2index[w])
    X = torch.LongTensor([index_seq])
    return X

def topic_classifier(text):
    state_dict = torch.load("model/NewsNet.pth")
    model = NewsNet(**state_dict['Param'])
    model.load_state_dict(state_dict['NewsNet'])
    result = model.predict(text, state_dict['stoi'], state_dict['labelMap'])
    return None, result