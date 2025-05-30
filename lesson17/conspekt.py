# Обнаружение аномалий в данных
# Метод главных компонент
# - уменьшим размерность данных
# - восстанановим размерность данных
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from fastai.text.all import *

df = pd.read_csv("./lesson17/data/creditcard.csv")

legit = df[df["Class"] == 0]
fraud = df[df["Class"] == 1]

legit = legit.drop(["Class", "Time"], axis=1)
fraud = fraud.drop(["Class", "Time"], axis=1)
print(legit.shape)
print(fraud.shape)

pca = PCA(n_components=26, random_state=0)
legit_pca = pd.DataFrame(pca.fit_transform(legit), index=legit.index)
fraud_pca = pd.DataFrame(pca.fit_transform(fraud), index=fraud.index)

print(legit_pca.shape)
print(fraud_pca.shape)

legit_restore = pd.DataFrame(pca.inverse_transform(legit_pca), index=legit_pca.index)
fraud_restore = pd.DataFrame(pca.inverse_transform(fraud_pca), index=fraud_pca.index)

print(legit_restore.shape)
print(fraud_restore.shape)


def anomaly_calc(original, restored):
    loss = np.sum((np.array(original) - np.array(restored)) ** 2, axis=1)
    return pd.Series(data=loss, index=original.index)


legit_calc = anomaly_calc(legit, legit_restore)
fraud_calc = anomaly_calc(fraud, fraud_restore)


# fig, ax = plt.subplots(1, 2, sharex="col", sharey="row")
# ax[0].plot(legit_calc)
# ax[1].plot(fraud_calc)

# plt.show()

th = 100
legit_TRUE = legit_calc[legit_calc < th].count()
legit_FALSE = legit_calc[legit_calc >= th].count()

print(legit_TRUE)
print(legit_FALSE)

fraud_TRUE = fraud_calc[fraud_calc < th].count()
fraud_FALSE = fraud_calc[fraud_calc >= th].count()

print(fraud_TRUE)
print(fraud_FALSE)


# NLP - Natural Language Processing
# Языковая модель позволяет предсказать следующее слово зная предыдущие. Метки не требуются, Но нужно очень много текста.
# Метки получаются автоматически из данных
#


path = untar_data(URLs.HUMAN_NUMBERS)

# print(path.ls())

lines = L()
with open("/Users/askoritan/.fastai/data/human_numbers/valid.txt") as f:
    lines += L(*f.readlines())

text = " ".join([l.strip() for l in lines])

tokens = text.split(" ")


vocab = L(*tokens).unique()

word2index = {w: i for i, w in enumerate(vocab)}
# print(word2index)

nums = L(word2index[w] for w in tokens)

# 1 Список из последовательности из трех слов
seq = L((tokens[i : i + 3], tokens[i + 3]) for i in range(0, len(tokens) - 4, 3))


seq = L((nums[i : i + 3], nums[i + 3]) for i in range(0, len(nums) - 4, 3))

seq = L((tensor(nums[i : i + 3]), (nums[i + 3])) for i in range(0, len(nums) - 4, 3))

bs = 64
cut = int(len(seq) * 0.8)
dls = DataLoaders.from_dsets(seq[cut:], seq[:cut], bs=bs, shuffle=False)


class Model1(Module):
    def __init__(self, vocab_size, n_hidden):
        self.i_h = nn.Embedding(vocab_size, n_hidden)
        self.h_h = nn.Linear(n_hidden, n_hidden)
        self.h_o = nn.Linear(n_hidden, vocab_size)

    def forward(self, x):
        # h = F.relu(self.h_h(self.i_h(x[:, 0])))
        # h = h + self.i_h(x[:, 1])
        # h = F.relu(self.h_h(h))  # h2
        # h = h + self.i_h(x[:, 2])
        # h = F.relu(self.h_h(h))  # h3
        h = 0
        for i in range(3):
            h = h + self.i_h(x[:, i])
            h = F.relu(self.h_h(h))

        return self.h_o(h)


learn = Learner(
    dls, Model1(len(vocab), bs), loss_func=F.cross_entropy, metrics=accuracy
)

learn.fit_one_cycle(4, 0.001)

n = 0
count = torch.zeros(len(vocab))
for x, y in dls.valid:
    n += y.shape[0]
    for i in range_of(vocab):
        count[i] += (y == 1).long().sum()


print(count)

index = torch.argmax(count)

print(index, vocab[index.item()], count[index].item() / n)
