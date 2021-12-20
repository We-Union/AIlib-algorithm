from typing import List, Union
import numpy as np
import json
from time import time
import torch

EPSILON = -3.14e100

def viterbi(A, B, pi, O):
    """
    Test it with:
        A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])

    B = np.array([
        [0.5, 0.5], [0.4, 0.6], [0.7, 0.3]
    ])

    pi = np.array([0.2, 0.4, 0.4])

    O = np.array([0, 1, 0])
    viterbi(A, B, pi, O)   
    """

    if isinstance(O, str):
        O = list(O)
    T = len(O)
    delta = np.zeros(shape=(T, A.shape[0]), dtype="float32")
    psi = np.zeros(shape=(T, A.shape[0]), dtype="int64")
    for i, o_i in enumerate(O):
        if i == 0:

            delta[0] = pi * B[..., O[0]]
            psi[0] = np.zeros(A.shape[0])
        else:
            cur_prod = delta[i - 1] * A.T
            psi[i] = np.argmax(cur_prod, axis=1)
            delta[i] = np.max(cur_prod, axis=1) * B[..., O[i]]
    print(delta)
    print(psi)
    return delta, psi

def normalise_dict(str_int_dict : dict, to_log : bool):
    count_sum = sum(str_int_dict.values())
    if count_sum == 0:
        if to_log:
            for key in str_int_dict:
                str_int_dict[key] = EPSILON
        return
    for key in str_int_dict:
        prob = str_int_dict[key] / count_sum
        if to_log:
            prob = np.log(prob) if prob > 0 else EPSILON
        str_int_dict[key] = prob

class ChineseSpliter(object):
    def __init__(self, log_prob=True) -> None:
        self.state = ['B', 'E', 'M', 'S']
        self.pi = {'B' : 0, 'E' : 0, 'M' : 0, 'S' : 0}
        self.A = {
            'B' : {'B' : 0, 'E' : 0, 'M' : 0, 'S' : 0}, 
            'E' : {'B' : 0, 'E' : 0, 'M' : 0, 'S' : 0}, 
            'M' : {'B' : 0, 'E' : 0, 'M' : 0, 'S' : 0}, 
            'S' : {'B' : 0, 'E' : 0, 'M' : 0, 'S' : 0}
        }                                                   # transition matrix
        self.B = {'B' : {}, 'E' : {}, 'M' : {}, 'S' : {}}   # emission matrix

        self.pi_vec = None
        self.A_mat = None

        self.log_prob = log_prob

    
    def B_col_array(self, o_i):
        b_list = []
        for state in self.state:
            if o_i not in self.B[state]:
                self.B[state][o_i] = 0
            b_list.append(self.B[state][o_i])
        return np.array(b_list)

    def word2BEMS(self, word):
        BEMS_list = []
        if len(word) in [0, 1]:
            BEMS_list.append('S')
        else:
            BEMS_list.append('B')
            BEMS_list.extend(['M'] * (len(word) - 2))
            BEMS_list.append('E')
        return BEMS_list
    

    def learn_from_word_list(self, word_list):
        tag_list = [''.join(self.word2BEMS(word)) for word in word_list]
        self.pi[tag_list[0][0]] += 1

        sentence = ''.join(word_list)
        tags = ''.join(tag_list)
        # update B
        for tag, ch in zip(tags, sentence):
            self.B[tag][ch] = self.B[tag].get(ch, 0) + 1
        # update A
        for i in range(1, len(tags)):
            self.A[tags[i - 1]][tags[i]] += 1


    def learn_from_file(self, file_str : str, encoding='utf-8'):
        for line in open(file_str, "r", encoding=encoding):
            line = line.strip()
            if line:
                self.learn_from_word_list(line.split())
        
        # normalise
        normalise_dict(self.pi, to_log=self.log_prob)
        for key in self.A:
            normalise_dict(self.A[key], to_log=self.log_prob)
        for key in self.B:
            normalise_dict(self.B[key], to_log=self.log_prob)
        
        # to constant
        self.pi_vec = np.array(list(self.pi.values()))
        self.A_mat = np.concatenate([np.array(list(self.A[key].values())).reshape(1, -1) 
                                    for key in self.A], axis=0)



    
    def cal_forward_prob(self, O : Union[str, List[str]]):
        if isinstance(O, str):
            O = list(O)
        T = len(O)
        alpha = np.zeros(shape=(T, len(self.state)), dtype="float32")

        for i, o_i in enumerate(O):
            if i == 0:
                if self.log_prob:
                    ...
                else:
                    alpha[0] = self.pi_vec * self.B_col_array(O[0])
            else:
                if self.log_prob:
                    ...
                else:
                    alpha[i] = alpha[i - 1] @ self.A_mat * self.B_col_array(O[i])
        return alpha
    
    def cal_backward_prob(self, O : Union[str, List[str]]):
        if isinstance(O, str):
            O = list(O)
    
    def viterbi(self, O):
        if isinstance(O, str):
            O = list(O)
        T = len(O)
        delta = np.zeros(shape=(T, len(self.state)), dtype="float32")
        psi = np.zeros(shape=(T, len(self.state)), dtype="int64")
        for i, o_i in enumerate(O):
            if i == 0:
                if self.log_prob:
                    delta[0] = self.pi_vec + self.B_col_array(O[0])
                else:
                    delta[0] = self.pi_vec * self.B_col_array(O[0])
                psi[0] = np.zeros(len(self.state))
            else:
                if self.log_prob:
                    cur_prod = delta[i - 1] + self.A_mat.T
                else:
                    cur_prod = delta[i - 1] * self.A_mat.T
                psi[i] = np.argmax(cur_prod, axis=1)
                delta[i] = np.max(cur_prod, axis=1) * self.B_col_array(O[i])
        
        return delta, psi
    
    
    def cut(self, O):
        delta, psi = self.viterbi(O)

        predict_state = [np.argmax(delta[-1])]
        for a in psi[::-1]:
            predict_state.append(a[predict_state[-1]])
        predict_state.pop()
        tags = [self.state[s] for s in predict_state[::-1]]
        split_result = []
        temp = ""
        for i, tag in enumerate(tags):
            if tag in ['B', 'M']:
                temp += O[i]
            elif tag == 'S':
                split_result.append(O[i])
            elif tag == 'E':
                temp += O[i]
                split_result.append(temp)
                temp = ""
        return split_result
    
    def save_model(self, path):
        state_dict = {
            "pi_vec" : self.pi_vec,
            "A_mat" : self.A_mat,
            "B" : self.B
        }

        torch.save(state_dict, path)
    
    def load_model(self, path):
        state_dict = torch.load(path)
        self.pi_vec = state_dict["pi_vec"]
        self.A_mat = state_dict["A_mat"]
        self.B = state_dict["B"]


def kanji_cut(text, spliter=" ",model_path="model/py_cut.pth"):
    model = ChineseSpliter(log_prob=False)
    model.load_model(model_path)
    result = []
    for sub_seq in text.split("。"):
        result += model.cut(text)
    result = spliter.join(result)
    return result