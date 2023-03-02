
#encoding=utf8

import os
import FukuML.Utility as utility
import FukuML.RidgeRegression as ridge_regression

input_train_data_file = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'library/iNDIEVOX-Dataset/dataset/valence_train.dataset')

cross_validator = utility.CrossValidator()

ridge_regression1 = ridge_regression.RidgeRegression()
ridge_regression1.load_train_data(input_train_data_file)
ridge_regression1.set_feature_transform('legendre', 2)
ridge_regression1.set_param(lambda_p=0)
ridge_regression2 = ridge_regression.RidgeRegression()
ridge_regression2.load_train_data(input_train_data_file)
ridge_regression2.set_feature_transform('legendre', 2)
ridge_regression2.set_param(lambda_p=0.01)

print("\n10 fold cross validation：")

cross_validator.add_model(ridge_regression1)
cross_validator.add_model(ridge_regression2)
avg_errors = cross_validator.excute()

print("\n各模型驗證平均錯誤：")
print(avg_errors)
print("\n最小平均錯誤值：")