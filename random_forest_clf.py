'''

Performs random forest classfication with k-fold cv
Expects data to be a .tar.gz file with the following structure

tar
│
└───fold0
│   │
│   └───train
│   │   │   file111.png
│   │   │   ...
│   │
│   └───val
│       │   file112.png
│       │   ...
│
└───fold1
    │   ...

'''


import pickle
import numpy as np
import pandas as pd
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn import neural_network, metrics, model_selection
from matplotlib import pyplot as plt
import tarfile
import shutil
from utils import autolabel


fold = 5
fold_lst = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']
tar_dir = '/home/jfeinst/Projects/resnet-files/features-binary.tar.gz'
tar_extract_path = '/home/jfeinst/Projects/resnet-files/'
tar_name = tar_dir.split('/')[-1].split('.')[0]
accuracy_lst = []
mathews_cc_lst = []


for i in range(fold):
    print('--Fold {}--'.format(i + 1))
    print('Extracting tarball...')


    # Untar tarball containing data
    with tarfile.open(tar_dir) as tar:
        subdir_and_files = [tarinfo for tarinfo in tar.getmembers() if
                            tarinfo.name.startswith(tar_name + '/' + fold_lst[i])]
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, members=subdir_and_files, path=tar_extract_path)


    train1path = tar_extract_path + tar_name + '/' + fold_lst[i] + '/train/1/*/*.pickle'
    train3path = tar_extract_path + tar_name + '/' + fold_lst[i] + '/train/3/*/*.pickle'
    test1path = tar_extract_path + tar_name + '/' + fold_lst[i] + '/val/1/*/*.pickle'
    test3path = tar_extract_path + tar_name + '/' + fold_lst[i] + '/val/3/*/*.pickle'
    train_list = [train1path, train3path]
    test_list = [test1path, test3path]


    # create training dataset for first class. Break after 930 because 2nd class is only 930 in size.
    count = 0
    class_list = [1.0, 3.0]
    print('Creating dataset...')
    # for x in range(len(class_list)):
    for dillpickle in glob.glob(train_list[0]):
        count += 1
        if count == 1:
            train1data = pickle.load(open(dillpickle, 'rb'))
            train1classes = np.array([class_list[0]])
        elif count == 930:
            break
        else:
            feature_vec = pickle.load(open(dillpickle, 'rb'))
            train1data = np.vstack((train1data,feature_vec))
            train1Class = np.array([class_list[0]])
            train1classes = np.vstack((train1classes, train1Class))


    # create training dataset for second class
    count = 0
    for dillpickle in glob.glob(train_list[1]):
        count += 1
        if count == 1:
            train2data = pickle.load(open(dillpickle, 'rb'))
            train2classes = np.array([class_list[1]])
        else:
            feature_vec = pickle.load(open(dillpickle, 'rb'))
            train2data = np.vstack((train2data,feature_vec))
            train2Class = np.array([class_list[1]])
            train2classes = np.vstack((train2classes, train2Class))


    # training datasets for each class created separately to make it easier to
    # manipulate one class's dataset to reduce overfitting


    train2datafivestack = np.vstack((train2data, train2data, train2data, train2data, train2data))
    train2classesfivestack = np.vstack((train2classes, train2classes, train2classes, train2classes, train2classes))


    print('Class 1 size: ', train1data.size, train1classes.size)
    print('Class 3 size: ', train2data.size, train2classes.size)


    traindata = np.vstack((train1data, train2data))
    trainclasses = np.vstack((train1classes, train2classes))


    # final datasets for training
    X_train = pd.DataFrame(traindata) # feature vector data
    Y_train = pd.DataFrame(trainclasses) # label data


    count = 0
    for x in range(len(class_list)):
        for dillpickle in glob.glob(test_list[x]):
            count += 1
            if count == 1 and x == 0:
                valdata = pickle.load(open(dillpickle, 'rb'))
                valclasses = np.array([class_list[x]])
            else:
                feature_vec = pickle.load(open(dillpickle, 'rb'))
                valdata = np.vstack((valdata,feature_vec))
                valClass = np.array([class_list[x]])
                valclasses = np.vstack((valclasses, valClass))


    # datasets for testing
    X_test = pd.DataFrame(valdata) # feature vector data
    Y_test = pd.DataFrame(valclasses) # label data


    print('Creating classifier...')
    clf = RandomForestClassifier(n_estimators=100) # defining the classfiers
    mlp = neural_network.MLPClassifier()


    print('Fitting forest...')
    clf.fit(X_train,Y_train.values.ravel()) # training random forest on training datasets
    y_pred_forest = clf.predict(X_test) # getting predictions from trained model using test datasets
    confusion_matrix_forest = metrics.confusion_matrix(Y_test, y_pred_forest) # create confusion matrix with predictions
    mathew_cc_forest = metrics.matthews_corrcoef(Y_test, y_pred_forest) # calculate mathews correlation coefficient
    accuracy = metrics.accuracy_score(Y_test, y_pred_forest) # calculate accuracy

    print('Confusion matrix: ', confusion_matrix_forest)
    print("Accuracy:", accuracy)
    print('Matthews correlation coefficient: ', mathew_cc_forest)


    # multilayer perceptron code. stats are not plotted at the end
    print('Fitting mlp...')
    mlp.fit(X_train, Y_train.values.ravel())
    y_pred_mlp = mlp.predict(X_test)
    confusion_matrix_mlp = metrics.confusion_matrix(Y_test, y_pred_mlp)
    mathew_cc_mlp = metrics.matthews_corrcoef(Y_test, y_pred_mlp)

    print('Confusion matrix: ', confusion_matrix_mlp)
    print("Accuracy:", metrics.accuracy_score(Y_test, y_pred_mlp))
    print('Matthews correlation coefficient: ', mathew_cc_mlp)


    # save random forest stats for final plotting
    accuracy_lst.append(accuracy)
    mathews_cc_lst.append(mathew_cc_forest)


    # delete directory to free space for next fold
    shutil.rmtree(tar_extract_path + tar_name)


# Plotting accuracy and mathews cc
labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, accuracy_lst, width, label='acc')
rects2 = ax.bar(x + width/2, mathews_cc_lst, width, label='mathew')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Accuracy and Mathews Correlation Coefficient')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

autolabel(rects1)
autolabel(rects2)

plt.ylim([0,1])
fig.tight_layout()
plt.savefig('randomforest-acc-mathewscc.png', dpi=2000, bbox_inches="tight")
# plt.show()
