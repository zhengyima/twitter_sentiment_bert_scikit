import numpy as np
import pandas
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="the model for classify, svm|rf|xb", type = str, default= "xb")
args = parser.parse_args()
model = args.model

def divide_dataset(X,Y):
    shuffle_list = [i for i in range(0,len(X))]
    np.random.shuffle(shuffle_list)
    X_train = [ X[shuffle_list[i]] for i in range(int(len(shuffle_list) * 0.8))]
    Y_train = [ Y[shuffle_list[i]] for i in range(int(len(shuffle_list) * 0.8))]

    X_dev = [ X[shuffle_list[len(X_train)+j]] for j in range(int(len(shuffle_list) * 0.1)) ]
    Y_dev = [ Y[shuffle_list[len(X_train)+j]] for j in range(int(len(shuffle_list) * 0.1)) ]
    
    # print(len(X_train)+len(X_dev))
    # print(len(shuffle_list))
    # print(len(Y))
    # print(len(X))
    X_test = [ X[shuffle_list[k]] for k in range(len(X_train)+len(X_dev),len(shuffle_list)) ]
    Y_test = [ Y[shuffle_list[k]] for k in range(len(X_train)+len(X_dev),len(shuffle_list)) ]

    return X_train,Y_train,X_dev,Y_dev,X_test,Y_test
        


def SVM_MODEL(X_train,Y_train,X_dev,Y_dev,X_end,Y_end):
    clf = svm.SVC(C=5, gamma=0.05,max_iter=-1)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_dev)
    print(Y_pred)
    precision = sum(Y_pred == Y_dev)/len(Y_dev)
    print('dev precision: ', precision)
    Y_pred = clf.predict(X_test)
    precision_test = sum(Y_pred == Y_test)/len(Y_test)
    print('test precision: ', precision_test)

    # print(shuffle_list)

def RF_MODEL(X_train,Y_train,X_dev,Y_dev,X_test,Y_test):
    clf = RandomForestClassifier(n_estimators=500, max_depth=32,random_state=8)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_dev)
    precision = sum(Y_pred == Y_dev)/len(Y_dev)
    print('dev precision: ', precision)
    Y_pred = clf.predict(X_test)
    precision_test = sum(Y_pred == Y_test)/len(Y_test)
    print('test precision: ', precision_test)


def XGBOOST_MODEL(X_train,Y_train,X_dev,Y_dev,X_test,Y_test):
    # clf = RandomForestClassifier(n_estimators=500, max_depth=32,random_state=8)
    clf = xgb.XGBClassifier(objective="multi:softprob",random_state=42,learning_rate=0.05,max_depth=8,n_estimators=100,subsample=0.5,colsample_bytree=0.5)
    clf.fit(np.array(X_train), np.array(Y_train))
    Y_pred = clf.predict(np.array(X_dev))
    precision = sum(Y_pred == Y_dev)/len(Y_dev)
    print('dev precision: ', precision)
    Y_pred = clf.predict(np.array(X_test))
    precision_test = sum(Y_pred == Y_test)/len(Y_test)
    print('test precision: ', precision_test)


if __name__ == '__main__':
    
    SENTIMENT_NAME_DIC = {'neutral':0,'positive':1,'negative':-1}
    Tweets = pandas.read_csv("./data/tweets.csv")
    airline_sentiments = Tweets['airline_sentiment'].tolist()
    Y = [SENTIMENT_NAME_DIC[air_sentiment] for air_sentiment in airline_sentiments]
    X = np.load("textvecs.npy")
    X_train,Y_train,X_dev,Y_dev,X_test,Y_test = divide_dataset(X,Y)

    if model == "svm":
        print('start training...')
        SVM_MODEL(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)
    elif model == "rf":
        print('start training...')
        RF_MODEL(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)
    elif model == "xb":
        print('start training...')
        XGBOOST_MODEL(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)
    else:
        print("default using xgboost")
        XGBOOST_MODEL(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)


