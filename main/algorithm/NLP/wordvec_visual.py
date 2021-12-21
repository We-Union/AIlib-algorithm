import os
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap

from main.algorithm.NLP.topic_classifier import NewsNet
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv

plt.rcParams["font.sans-serif"] = ["SimHei"] 
plt.rcParams["axes.unicode_minus"] = False   

all_models = {
    "pca" : PCA,
    "tsne" : TSNE,
    "mds" : MDS,
    "isomap" : Isomap
}

def visual_wordvec(text, decomposition_method="pca", s=120, alpha=0.9, fontsize=12, height=10, width=12):
    words = text.split()
    state_dict = torch.load("model/NewsNet.pth")
    model = NewsNet(**state_dict['Param'])
    model.load_state_dict(state_dict['NewsNet'])
    # model = 

    embedding_matrix = model.embedding.weight.detach().numpy()
    print(embedding_matrix.shape)

    model = all_models[decomposition_method.lower()](n_components=2)
    embedding_matrix = model.fit_transform(embedding_matrix)
    print(embedding_matrix.shape)

    index_seq = []
    for w in words:
        if w in state_dict['stoi']:
            index_seq.append(state_dict['stoi'][w])
    
    if len(index_seq) == 0:
        return 6012, None
    temp_file = "temp.png"

    plt.figure(figsize=(width, height))
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    for index in index_seq:
        point = embedding_matrix[index]
        plt.scatter(point[0], point[1], s=s, alpha=alpha)
        plt.text(point[0], point[1], state_dict['itos'][index], fontdict={"fontsize" : fontsize})
    
    plt.grid() 
    plt.savefig(temp_file)

    img = cv.imread(temp_file)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    os.remove(temp_file)
    return img, "分析成功"