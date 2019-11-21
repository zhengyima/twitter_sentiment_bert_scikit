from bert_serving.client import BertClient
import pandas
import numpy as np

bc = BertClient("183.174.228.116")
Tweet= pandas.read_csv("./data/tweets.csv")
# print(Tweet['text'][2001])
# print(Tweet['text'].tolist())

vecs = bc.encode(Tweet['text'].tolist())
vecs = np.array(vecs)
np.save("textvecs.npy",vecs)

# print(bc.encode(['second do it']))