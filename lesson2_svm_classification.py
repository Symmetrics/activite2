#encoding=utf8

import os
import FukuML.Utility as utility
import FukuML.SupportVectorMachine as svm

input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'library/iNDIEVOX-Dataset/dataset/emotion_combine_song_train.dataset')

cross_validator = utility.CrossValidator()

svm_mc1 = svm.MultiClassifier()
svm_mc1.load_train_data(input_train_data_file)
svm_mc1.set_param(svm_kernel='soft_gaussian_kernel', C=1)
svm_mc2 = svm.MultiClassifier()
svm_mc2.load_train_data(input_train_data_file)
svm_mc2.set_param(svm_kernel='soft_gaussian_kernel', C=10)
svm_mc3 = svm