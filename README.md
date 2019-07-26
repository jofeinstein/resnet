# ResNet

## Description
Contains files that use ResNet architecture with different hyper-parameters.

## Dependencies
* Python 3.6.8
* Numpy 1.16.2
* Pytorch 1.1 (if using GPU acceleration, use CUDA version)
* Matplotlib 3.0.3
* Torchvision 0.4
* Scikit-image 0.15.0
* Scikit-learn 0.21.0

## Files
* freeze_layers_train.py: trains ResNet while having the ability to freeze layers of the network. User has the option to use weighted loss functions and the weighted random sampler. Expects a tarfile with the structure seen below.
```
usage: python freeze_layers.py -tar_extract_path /var/scratch/user/ -tar_dir /work/user/resnet/test.tar.gz -epoch 100 -learning_rate 0.001  
```
```
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
```

* utils.py: contains various functions critical to ResNet and random forest. It contains:
  * two training functions. One with validation and one without
  * functions to initialize networks
  * function used to calculate weights
  * various plotting functions 

* kfold_cv_train.py: Same as freeze_layers_train.py except it will iterate through a user specified amount of folds for cross validation.
```
usage: python kfold_cv_train.py -tar_extract_path /var/scratch/user/ -tar_dir /work/user/resnet/test.tar.gz -epoch 100 -learning_rate 0.001 
```

* random_forest_clf.py: Run random forest classifier and multilayer perception on a binary dataset. Expects same directory structure as above.
```
```

* no_validation_train.py: Train a ResNet model with no validation. Expects data to be in a directory, although data can be several layers deep.
```
usage: python no_validation_train.py -epoch 100 -data_dir /work/user/resnet/data -model_file ./log/modelname.pt
```