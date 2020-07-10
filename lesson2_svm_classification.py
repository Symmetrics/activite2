#encoding=utf8

import os
import FukuML.Utility as utility
import FukuML.SupportVectorMachine as svm

input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'library/iNDIEVOX-Dataset/dataset/emotion_combine_song_train.dataset')

cross_validator = utility.CrossValidator()

svm_mc1 = svm.MultiClassifier