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
svm_mc3 = svm.MultiClassifier()
svm_mc3.load_train_data(input_train_data_file)
svm_mc3.set_param(svm_kernel='soft_gaussian_kernel', C=100)

print("\n10 fold cross validation：")

cross_validator.add_model(svm_mc1)
cross_validator.add_model(svm_mc2)
cross_validator.add_model(svm_mc3)
avg_errors = cross_validator.excute()

print("\n各模型驗證平均錯誤：")
print(avg_errors)
print("\n最小平均錯誤率：")
print(cross_validator.get_min_avg_error())

print("\n取得最佳模型：")
best_model = cross_validator.get_best_model()
print(best_model)

best_model.init_W()
best_model.train()

future_data = '0.0278168491121 0.0153081289758 3.03709327891 0.10743602255 0.174952836962 0.134500894777 0.00925739148295 0.0366942618855 -26.8290719952 2.69594124651 -0.203409679263 0.342470125679 0.0519006783931 0.0648784790254 -0.026167369055 -0.0277357874536 -0.0510789186296 0.0561642315562 0.0752938427254 0.153650035253 0.127605292534 0.00863423677803 0.00152184278569 0.0204106055354 0.00385907385983 0.0