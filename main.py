import random
import time

import nltk
import numpy as np
import os
from nltk.stem import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt


def main():
    dir_path = 'sentiment labelled sentences'
    # file_name = 'amazon_cells_labelled.txt'
    file_names = os.listdir(dir_path)
    file_names.remove('readme.txt')
    file_paths = [os.path.join(dir_path, file_name) for file_name in file_names]

    stemmer = PorterStemmer()
    dic = extract_dic(file_paths, stemmer)
    vocabulary = extract_vocabulary(dic)

    X, y = extract_feature(file_paths, stemmer, vocabulary)

    X = tf_idf_feature(X)

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
    X, y = X_test, y_test
    # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=43)

    base_acc_mat = []
    acc_mat_1 = []
    acc_mat_2 = []
    times = []

    x = X[0]
    x = x.reshape([1, len(x)])

    start = time.time()
    base_acc = jaccard_similarity(X, x)
    scal_time = time.time()
    times.append(scal_time-start)

    k = 16
    start = time.time()
    acc = jaccard_similarity_hash(k_mini_hash(X, k), k_mini_hash(x, k))
    scal_time = time.time()
    mle1 = np.square(acc - base_acc).mean()
    times.append(scal_time-start)

    k = 128
    start = time.time()
    acc2 = jaccard_similarity_hash(k_mini_hash(X, k), k_mini_hash(x, k))
    scal_time = time.time()
    mle2 = np.square(acc2 - base_acc).mean()
    times.append(scal_time-start)

    print(times)
    print(mle1)
    print(mle2)

    # start = time.time()
    # for x in X:
    #     x = x.reshape([1, len(x)])
    #     base_acc = jaccard_similarity(X, x)
    #     base_acc_mat.append(base_acc)
    # scal_time = time.time()
    # times.append(scal_time-start)
    # base_acc_mat = np.array(base_acc_mat)

    # start = time.time()
    # for x in X:
    #     k = 16
    #     x = x.reshape([1, len(x)])
    #     acc = jaccard_similarity_hash(k_mini_hash(X, k), k_mini_hash(x, k))
    #     acc_mat_1.append(acc)
    #
    # scal_time = time.time()
    # times.append(scal_time-start)
    # acc_mat_1 = np.array(acc)
    # evl = np.square(acc_mat_1 - base_acc_mat).mean()
    #
    # start = time.time()
    # for x in X:
    #     k = 128
    #     x = x.reshape([1, len(x)])
    #     acc = jaccard_similarity_hash(k_mini_hash(X, k), k_mini_hash(x, k))
    #     acc_mat_2.append(acc)
    # scal_time = time.time()
    # times.append(scal_time-start)
    # acc_mat_2 = np.array(acc)
    # evl2 = np.square(acc_mat_2 - base_acc_mat).mean()
    #
    #
    # print(times)
    # print(evl)
    # print(evl2)


def jaccard_similarity(X1, X2):
    return np.count_nonzero((X1 > 0) & (X2 > 0), axis=1) / np.count_nonzero(X1 + X2, axis=1)

def jaccard_similarity_hash(X1, X2):
    return np.count_nonzero(X1 == X2, axis=0) / X1.shape[0]


def signature(X, permutation):
    X_ = np.transpose(X)
    X_ = X_[permutation]
    mask = X_>0
    s = np.where(mask.any(axis=0), mask.argmax(axis=0), -1)
    return s

def k_mini_hash(X, k):
    s = []
    sd = np.random.RandomState(seed=42).randint(50, size=k)
    for i in range(k):
        permutation = np.random.RandomState(seed=sd[i]).permutation(X.shape[1])
        s.append(signature(X, permutation))
    return np.array(s)

    # method_list = {'std_knn': KNeighborsClassifier(n_neighbors=5), 'my_knn': KNN(5),
    #                'Classifier': DecisionTreeClassifier(random_state=0),
    #                'SVM': make_pipeline(StandardScaler(), SVC(gamma='auto'))}
    #
    # performance_list = {'scal_time':[], 'pred_time':[], 'test_acc':[], 'train_acc':[], 'valid_acc':[]}
    # for method in method_list:
    #     clf = method_list[method]
    #     print(f'for {method}')
    #     start = time.time()
    #     clf.fit(X_train, y_train)
    #     scal_time = time.time()
    #     print(f'scal_time: {scal_time-start}')
    #     performance_list['scal_time'].append(scal_time-start)
    #
    #     y_pred = clf.predict(X_test)
    #     pred_time = time.time()
    #     print(f'pred_time: {pred_time-scal_time}')
    #     performance_list['pred_time'].append(pred_time-start)
    #
    #     acc = accuracy_score(y_test, y_pred)
    #     cm = confusion_matrix(y_test, y_pred, normalize='true')
    #     performance_list['test_acc'].append(acc)
    #     # performance_list['cm'].append(cm)
    #
    #     y_pred = clf.predict(X_train)
    #     acc = accuracy_score(y_train, y_pred)
    #     performance_list['train_acc'].append(acc)
    #
    #     y_pred = clf.predict(X_valid)
    #     acc = accuracy_score(y_valid, y_pred)
    #     performance_list['valid_acc'].append(acc)
    #
    # for k in performance_list:
    #     fig, ax = plt.subplots()
    #     methods = list(method_list.keys())
    #     p = performance_list[k]
    #     ax.bar(methods, p)
    #     ax.set_ylabel('acc/time(in ms)')
    #     ax.set_title(k)
    # plt.show()
    # print(performance_list)


def tf_idf_feature(feature):
    training_feature = np.array(feature)
    tf = np.sum(training_feature, 0)
    n_sentence = np.count_nonzero(training_feature, axis=0)
    # for feature in training_feature:
    # avoid / 0
    n_sentence[n_sentence == 0] = 100000
    idf = np.log(len(training_feature) / n_sentence)
    tf_idf = np.multiply(tf, idf)
    # sort in descending , adding -
    sort_i = np.argsort(-tf_idf)
    # test 1: using condition
    condition = sort_i < 512
    # extracted_feature = np.array([np.extract(condition, line) for line in training_feature])
    extracted_feature = np.compress(condition, training_feature, axis=1)
    return extracted_feature


def randomlize(feature):
    n = len(feature)
    div = {'train': 0.7, 'test': 0.15, 'validate': 0.15}
    train_feature, test_feature, valid_feature = [], [], []
    while len(feature) > 0:
        i = random.randrange(0, len(feature))
        if len(train_feature) < n * div['train']:
            train_feature.append(feature.pop(i))
        elif len(test_feature) < n * div['test']:
            test_feature.append(feature.pop(i))
        else:
            valid_feature.append(feature.pop(i))
    print(len(train_feature), len(test_feature), len(valid_feature))
    return train_feature


def k_means(feature, k=2, epoch=100):
    centers_i = [np.random.randint(0, len(feature)) for _ in range(k)]
    centers = feature[centers_i]
    for _ in range(epoch):
        distances = np.linalg.norm(feature - centers[:, np.newaxis], axis=-1)
        groups = np.argmin(distances, axis=0)
        for i in range(k):
            group_mask = (groups == i)
            if np.any(group_mask):
                centers[i] = np.mean(feature[group_mask], axis=0)
    return groups, centers


# def naive_bayes(feature):
class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    # def predict(self, X_):
    #     X = self.X
    #     dis = np.linalg.norm(X-X_[:, np.newaxis], axis=-1)
    #     indexes = np.argsort(dis, axis=0)[:, self.k]
    #     target_labels = self.y[indexes]
    #     counts = np.apply_along_axis(np.bincount, -1, target_labels)
    #     result = np.argmax(counts, axis=-1)
    #     return result

    def predict(self, X_):
        y_pred = np.apply_along_axis(self.predict_, 1, X_)
        return y_pred

    def predict_(self, X_):
        X = self.X
        dis = np.linalg.norm(X - X_, axis=-1)
        indexes = np.argsort(dis)[:self.k]
        target_labels = self.y[indexes]
        counts = np.bincount(target_labels)
        result = np.argmax(counts)
        return result




class KNN_jaccord:
    def __init__(self, k=1):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_):
        y_pred = np.apply_along_axis(self.predict_, 1, X_)
        return y_pred

    def predict_(self, X_):
        X = self.X
        dis = self.jaccard_similarity(X, X_)
        indexes = np.argsort(dis)[:self.k]
        target_labels = self.y[indexes]
        counts = np.bincount(target_labels)
        result = np.argmax(counts)
        return result

    def jaccard_similarity(self, X1, X2):
        np.count_nonzero((X1 > 0) == (X2 > 0), axis=1) / np.count_nonzero(X1 + X2)

class KNN_min_hash:
    def __init__(self, k=1):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_):
        y_pred = np.apply_along_axis(self.predict_, 1, X_)
        return y_pred

    def predict_(self, X_):
        X = self.X
        dis = self.jaccard_similarity(X, X_)
        indexes = np.argsort(dis)[:self.k]
        target_labels = self.y[indexes]
        counts = np.bincount(target_labels)
        result = np.argmax(counts)
        return result

    def signature(self, X, permutation):
        X_ = np.transpose(X)
        X_ = X_[permutation]
        mask = X_>0
        s = np.where(mask.any(axis=0), mask.argmax(axis=0), -1)
        return s

    def similarity(self, X, k):
        s = []
        seed = np.random.randint(50, size=k)
        for i in range(k):
            permutation = np.random.RandomState(seed=seed[i]).permutation(len(X))
            s.append(self.signature(X, permutation))








def test_knn():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    knn = KNN(1)
    knn.train(X, y)
    knn.eval(np.array([[1.1], [2.1]]), np.array([0, 1]))
    # knn.predict(np.array([1.1]))


def extract_feature(file_paths, stemmer, vocabulary):
    n = len(vocabulary)
    # feature !
    feature = []
    labels = []
    sample_sentence = []
    sample_index = 0
    # construct feature matrix
    for file_path in file_paths:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if sample_index < 5:
                    sample_sentence.append([sample_index, line])
                    sample_index += 1
                sentence_feature = [0 for _ in range(n)]
                sentence, label = line.split("\t")
                words = nltk.word_tokenize(sentence)
                for word in words:
                    word = stemmer.stem(word)
                    i = vocabulary.index(word)
                    sentence_feature[i] += 1
                feature.append(sentence_feature)
                labels.append(int(label))
    # print("feature size: ", len(feature), "*", len(feature[0]))
    # for i, line in sample_sentence:
    #     print("sample line:", line, end="")
    #     print("sample feature:", feature[i])
    return feature, labels


def extract_vocabulary(dic):
    # initiate feature matrix
    vocabulary = []
    # where ft,d is simply the frequency of word t in document d.
    for key in dic.keys():
        vocabulary.append(key)
    return vocabulary


def extract_dic(file_paths, stemmer):
    # get word frequency dic{word->frequency} to get n of matrix
    dic = {}
    for file_path in file_paths:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                sentence, label = line.split("\t")
                words = nltk.word_tokenize(sentence)
                for word in words:
                    word = stemmer.stem(word)
                    if word in dic:
                        dic[word] += 1
                    else:
                        dic[word] = 1
    return dic


if __name__ == '__main__':
    main()
    # test_knn()
