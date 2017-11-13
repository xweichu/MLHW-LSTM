import numpy as np
from sklearn import metrics
import csv
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def data_process(filename):
    raw_data = []
    with open(filename, 'r',encoding='utf8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            temp = [row[0]]
            line = row[1].split('https')
            temp.append(line[0])
            raw_data.append(temp)
        raw_data.pop(0)
    return raw_data

def textTf_idf(train_data,test_data):
    # tf-idf
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    #my_stop_words = text.ENGLISH_STOP_WORDS  # .union(["book"])
    # calculate the frequence of each word in a given document
    #vectorizer = CountVectorizer(stop_words=my_stop_words, decode_error='ignore')
    vectorizer = CountVectorizer(decode_error='ignore')

    counts = vectorizer.fit_transform(train_data[:,1]).toarray()
    transformer = TfidfTransformer()

    X_term = transformer.fit_transform(counts).toarray()
    labels = []
    for i in range(len(train_data)):
        if train_data[i][0]=='realDonaldTrump':
            labels.append(1)
        else:
            labels.append(0)#-1
    #test

    test_datax = test_data[:,1]
    counts1 = vectorizer.transform(test_datax).toarray()
    tst_X = transformer.transform(counts1).toarray()
    return X_term,labels,tst_X


def loadGloveModel(gloveFile):
    f = open(gloveFile, 'r',encoding='utf8')
    wordlist = []
    vectors = []
    for line in f:
        splitLine = line.split()
        wordlist.append(splitLine[0])
        vectors.append([float(val) for val in splitLine[1:]])
    vectors = np.array(vectors)
    return wordlist, vectors


def svm_cross_validation_tune(x_term, labels):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    params = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, params, n_jobs = 1, verbose=1)
    grid_search.fit(x_term, labels)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(x_term, labels)
    return model
def svm_CV(x_term, labels):
    params = {'C': [1e-2, 1e-1, 1, 10, 100], 'gamma': [0.001, 0.0001]}
    clf = GridSearchCV(SVC(), params, cv=5,scoring='precision')
    clf.fit(x_term,labels)
    means = clf.cv_results_['mean_test_score']

    return clf,means
#cross validation???
# def getnextBatchData():
#     kf = KFold(n_splits=10)
#     kf.get_n_splits(X)
#     KFold(n_splits=10, random_state=None, shuffle=False)
#loss function ??

#train-----------------------------------------------------------------

train_data = data_process('train.csv')
test_data = data_process('test.csv')
x_term,labels,tst_X = textTf_idf(train_data,test_data)

# np.save('train_x',x_term)
# np.save('lables',labels)
# np.save('tst_X',tst_X)
# x_term = np.load('train_x.npy')
# labels = np.load('lables.npy')
# tst_X = np.load('tst_X.npy')

model,means = svm_CV(x_term, labels)
labels, y_pred = labels, model.predict(x_term)
print(classification_report(labels, y_pred))
print("max means: %f" %max(means))
#test------------------------------------------------------------------
results = model.predict(tst_X)
print(results)
csvfile = open('pred.csv', 'w')
spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(['id', 'realDonaldTrump', 'HillaryClinton'])
for i in range(len(tst_X)):
    if results[i]==1:
        spamwriter.writerow([i, results[i],0])
    else:
        spamwriter.writerow([i,0, 1-results[i]])
predict1 = model.predict(x_term)
accuracy = metrics.accuracy_score(labels, predict1)
print("accuracy:%f"%accuracy)
#test code
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
# clf = SVC()
# clf.fit(X, y)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# print(clf.predict([[-0.8, -1],[0.2,4],[0.5,3]]))
