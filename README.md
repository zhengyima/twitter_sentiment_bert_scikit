# twitter_sentiment_bert_scikit
Twitter US Airline数据集情感分析（sentiment Analysis），使用Bert Sentence encoding作为特征，实现了SVM、XGBoost、RandomForest（随机森林）等若干分类算法，做了交叉验证。


# Preinstallation

We run the project in a **Python 3** Environment, and we recommend you use **Anaconda 3** to install the required package with the scripts below. Of course you can use **pip** to install.


```
 conda create -n tweet_sentiment -c anaconda python=3.7 numpy scikit-learn xgboost pandas tensorflow tensorflow-gpu

 conda activate tweet_sentiment

 pip install bert-serving-server # bert-service-server

 pip install bert-serving-client # bert-service-client

```

make sure **Tensorflow >= 1.10**

# Download BERT Model and Start BERT-Service

Go to [Bert](https://github.com/google-research/bert#pre-trained-models) and download a model. 

In this project we choose [BERT-Large, Uncased (WWM)](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)

Download the model , unzip it, and start the bert-service. You can also refer to [bert-as-service](https://github.com/hanxiao/bert-as-service), but in this project you only need to run the scripts below to achieve our goal.

```
 mkdir /tmp/bert_models

 unzip -d /tmp/bert_models/wwm_uncased_L-24_H-1024_A-16.zip # or another model zip file you downlaod

 bert-serving-start -model_dir /tmp/bert_models/wwm_uncased_L-24_H-1024_A-16/ -num_worker=4 -max_seq_len 256

```

# Launch the Sentiment Classification Script

**Open another terminal or screen while running the BERT-Service**

```
 conda activate tweet_sentiment

 python gen_vec.py # generate the sentence vectors and save to the npy file

 python model.py -d svm # svm model

 python model.py -d rf # random forest model

 python model.py -d xb # xgboost model

```


# RESULT

SVM Precision: 88%

Random Forest Precision:76%

XGBoost Precision:79%
