import numpy as np
from sklearn import svm
import random
from sklearn.cross_validation import KFold

def make_np_array_XY(xy):
    #print "make_np_array_XY()"
    a = np.array(xy)
    x = a[:,0:-1]
    y = a[:,-1]
    return x,y

def main3():
    print("\n\n\n\nSVM::")
    XY = []
    with open("spambase.csv") as f:
        for line in f:
            words = line.split(",")
            XY.append([float(w) for w in words])

    #XY_train= [[542, 34, 0.06273062730627306, 104, 0, 1], ....]
    #XY_test= [[758, 49, 0.06464379947229551, 133, 0, 1], ....]

    #k_fold = KFold(n=len(XY), n_folds=6)

    percent = 0.90

    XY_train = []
    XY_test = []
    num = int(percent * len(XY))
    print(num)
    count = 0
    tr = []
    while(1):
    #tr = [random.randrange(1,len(XY)+1) for j in range(num)]
      x = random.randrange(1,len(XY)+1)
      if x not in tr:
        tr.append(x)
        count = count + 1
        if count == num:
           break

    for i in tr:
        XY_train.append(XY[i-1])
    #print tr
    #print "len of tr=",len(tr)
    #print "len of set of tr=",len(set(tr))
    #print len(set(range(1,len(XY)+1)) - set(tr))
    for i in set(range(1,len(XY)+1)) - set(tr):
        XY_test.append(XY[i-1])

    X_train, Y_train = make_np_array_XY(XY_train)
    X_test, Y_test = make_np_array_XY(XY_test)

    print("total email =  ",len(XY))
    print("train set =  ",len(X_train),", test set = ",len(X_test))
    # train set
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)

    Y_predict = svc.predict(X_test)

    test_size = len(Y_test)
    score = 0

    TP,TN,FP,FN=0,0,0,0
    for i in range(test_size):
        if Y_predict[i] == Y_test[i]:
            if Y_predict[i] == 0:
                TP = TP + 1
            else:
                TN = TN + 1
            score += 1
        else:
            if Y_predict[i] == 0:
                FP = FP + 1
            else:
                FN = FN + 1

    
    print('[TP=',TP,",TN=",TN)
    print('FN=',FN,",FP=",FP,"]")
    print('Got %s out of %s' %(score, test_size))
    print('accuracy = ',float(score) / test_size)
    print('accuracy = ',float(TP+TN)/test_size)
