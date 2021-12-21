import torch
from torch import nn
from main.algorithm.NLP.translate import zh2en


def sentence2tensor(sentence, word2index):
    import string 
    punc = string.punctuation
    for w in punc:
        sentence = sentence.replace(w, "")
    index_seq = []
    for w in sentence.split():
        w = w.lower()
        if w not in word2index:
            index_seq.append(word2index['<unk>'])
        else:
            index_seq.append(word2index[w])
    X = torch.LongTensor([index_seq])
    return X

class MoodNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):

        super(MoodNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer_dim,
                         batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x : [bacth, time_step, vocab_size]
        embeds = self.embedding(x)
        # embeds : [batch, time_step, embedding_dim]
        r_out, h_n = self.gru(embeds, None)
        # r_out : [batch, time_step, hidden_dim]
        out = self.fc1(r_out[:, -1, :])
        # out : [batch, time_step, output_dim]
        return out    

    def predict(self, x, word2index, out_dict_str=False):
        out = self(sentence2tensor(x, word2index))
        out = nn.Softmax(dim=1)(out)
        label = ['负面心情', '正面心情']
        pre_lab = out.argmax(1)[0].item()
        if out_dict_str:
            print(out)
            scores = [out[0][0].item(), out[0][1].item()]
            out_str = """
[score] 负面心情 : {}
        正面心情 : {}

[result]         : {}
            """.format(round(scores[0], 3), round(scores[1], 3), label[pre_lab])
            return out_str
        else:
            return label[pre_lab]

def detect_mood(text, out_dict_str=True):
    for w in text:
        if '\u4e00' <= w <= '\u9fff':
            _, text = zh2en(text)
            break
    
    param = torch.load("model/MoodNet.pth")
    word2index = param['stoi']

    model = MoodNet(**param["Param"])
    model.load_state_dict(param['MoodNet'])

    return None, model.predict(text, word2index, out_dict_str=out_dict_str)