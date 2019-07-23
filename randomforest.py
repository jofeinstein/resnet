import pickle
import numpy as np
import pandas as pd
import os
from os import listdir
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn import neural_network, metrics, model_selection
from matplotlib import pyplot as plt
import tarfile

fold

for i in range(fold):
    print('--Fold {}--'.format(i + 1))

    print('Extracting tarball...')

    # Untar tarball containing data
    with tarfile.open(tar_dir) as tar:
        subdir_and_files = [tarinfo for tarinfo in tar.getmembers() if
                            tarinfo.name.startswith(tar_name + '/' + fold_lst[i])]
        tar.extractall(members=subdir_and_files, path=tar_extract_path)

    train1path = '/home/jfeinst/Projects/resnet-files/features/train/1/*/*.pickle'
    train3path = '/home/jfeinst/Projects/resnet-files/features/train/3/*/*.pickle'
    test1path = '/home/jfeinst/Projects/resnet-files/features/val/1/*/*.pickle'
    test3path = '/home/jfeinst/Projects/resnet-files/features/val/3/*/*.pickle'
    train_list = [train1path, train3path]
    test_list = [test1path, test3path]

    count = 0
    class_list = [1.0, 3.0]
    print('Creating dataset...')
    for x in range(len(class_list)):
        for dillpickle in glob.glob(train_list[x]):
            count += 1
            if count == 1 and x == 0:
                traindata = pickle.load(open(dillpickle, 'rb'))
                trainclasses = np.array([class_list[x]])
            else:
                feature_vec = pickle.load(open(dillpickle, 'rb'))
                traindata = np.vstack((traindata,feature_vec))
                trainClass = np.array([class_list[x]])
                trainclasses = np.vstack((trainclasses, trainClass))

    X_train = pd.DataFrame(traindata)
    Y_train = pd.DataFrame(trainclasses)

    count = 0
    for x in range(len(class_list)):
        for dillpickle in glob.glob(train_list[x]):
            count += 1
            if count == 1 and x == 0:
                valdata = pickle.load(open(dillpickle, 'rb'))
                valclasses = np.array([class_list[x]])
            else:
                feature_vec = pickle.load(open(dillpickle, 'rb'))
                valdata = np.vstack((valdata,feature_vec))
                valClass = np.array([class_list[x]])
                valclasses = np.vstack((valclasses, valClass))

    X_test = pd.DataFrame(valdata)
    Y_test = pd.DataFrame(valclasses)

    print('Creating classifier...')
    clf = RandomForestClassifier(n_estimators=100)
    mlp = neural_network.MLPClassifier()

    print('Fitting forest...')
    clf.fit(X_train,Y_train.values.ravel())
    y_pred_forest = clf.predict(X_test)
    confusion_matrix_forest = metrics.confusion_matrix(Y_test, y_pred_forest)
    mathew_cc_forest = metrics.matthews_corrcoef(Y_test, y_pred_forest)

    print('Confusion matrix: ', confusion_matrix_forest)
    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred_forest))
    print('Matthews correlation coefficient: ', mathew_cc_forest)





'''
pred = model_selection.cross_val_predict(mlp, X_train, Y_train.values.ravel(), cv=5)
plt.scatter(pred, Y_test)
confusion_matrix = metrics.confusion_matrix(Y_train, pred)
print(confusion_matrix)
mathew_cc = metrics.matthews_corrcoef(Y_train, pred)
print(mathew_cc)



print('Fitting mlp...')
mlp.fit(X_train, Y_train.values.ravel())
y_pred_mlp = mlp.predict(X_test)
confusion_matrix_mlp = metrics.confusion_matrix(Y_test, y_pred_mlp)
mathew_cc_mlp = metrics.matthews_corrcoef(Y_test, y_pred_mlp)

print('Confusion matrix: ', confusion_matrix_mlp)
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred_mlp))
print('Matthews correlation coefficient: ', mathew_cc_mlp)'''