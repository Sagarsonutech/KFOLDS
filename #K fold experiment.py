import os
import numpy as np
os.system('cls')
print('************starting***********')
x = np.arange(1, 25).reshape(12,2)
y = np.array([0,1,1,0,1,0,0,1,0,0,1,0,]) 
print(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)
print('x_train : ')
print(x_train)
print('x_test : ')
print(x_test)
print('y_train : ')
print(y_train)
print('y_test : ')
print(y_test) 

from sklearn.model_selection import LeaveOneOut
kf_one = LeaveOneOut()
kf_one.get_n_splits(x)
# KFLD OF LEAVE ONE OUT
print('Output of - k fold with Leave-One-Out cross-validator \n')
for i, (train_index, test_index) in enumerate(kf_one.split(x)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
print('\n')

#REPEATING with k-fold by n_split param|


from sklearn.model_selection import KFold
k_fol_spl=6
kf = KFold(n_splits=k_fol_spl,shuffle=False)
print('k-fold n_splits used :',k_fol_spl)
print('Output of - k fold for n_split \n')
for i, (train_index, test_index) in enumerate(kf.split(x)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

os.system('cls')

#method 1 nparray
X = np.arange(1,55).reshape(27,2)
y = np.array([1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1])
X, y = make_classification(n_samples=100, n_features=9, n_informative=5, n_redundant=2, random_state=1)
#Number of informative, redundant and repeated features must sum to less than the number of total features
print('Input X is \n',X[0:4])
print('The corresponding output u is \n',y[0:4])

from sklearn.model_selection import LeaveOneOut
kf_one = LeaveOneOut()
kf_one.get_n_splits(X)
print('Output of - k fold with Leave-One-Out cross-validator \n') 
for training_index, testing_index in kf_one.split(X):
    X_train, X_test = X[training_index], X[testing_index,:]
    Y_train, Y_test = y[training_index] , y[testing_index]
    model_lr_leave_one_out = LogisticRegression()
    model_lr_leave_one_out.fit(X_train,Y_train)
    Y_pred = model_lr_leave_one_out.predict(X_test)
    acc = accuracy_score(Y_pred , Y_test)
    print( 'The accuracy found for (Leave-One-Out) in this fold is : ',acc)


#kfold parameter splits


X = np.arange(1,55).reshape(27,2)
y = np.array([1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1])
X, y = make_classification(n_samples=100, n_features=9, n_informative=5, n_redundant=2, random_state=1)
#Number of informative, redundant and repeated features must sum to less than the number of total features
print('Input X is \n',X[0:2])
print('The corresponding output u is \n',y[0:2])

from sklearn.model_selection import KFold
k_fol_spl = 6
kf = KFold(n_splits = k_fol_spl, shuffle = True)
print('k-fold n_splits used :', k_fol_spl)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index] , y[test_index]
    print( 'Total Test samples = (Total samples / k-fold) \n' , 6/k_fol_spl)
    print(f"Test Values   =\n",X_test)
    print(f"Train Values  = \n",X_train)


model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print( 'Predicted output in this fold ... ' , Y_pred)
print( 'But actual output in this fold was ... ' , Y_test)
acc = accuracy_score(Y_pred , Y_test)
acc*100
print( 'The accuracy found in this fold is : ',acc)

#stratified kfold method
from sklearn.model_selection import StratifiedKFold
#create stratified kfold obejcts 
strati_fold = StratifiedKFold(n_splits= k_fol_spl, shuffle= True)
print("\n !!!!!!Starting skf calculations now !!!!!")
print('skf-fold n_splits used :',k_fol_spl)

for i, (train_index, test_index) in enumerate(strati_fold.split(X,y)):
    print(f"Fold {i}:")
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index] , y[test_index] 
    print( 'Total Test samples = (Total samples / k-fold) \n' , 22/k_fol_spl)
    print(f"Test Values  =\n",X_test)
    print(f"Train Values = \n",X_train)
    model_skf = LogisticRegression()
    model_skf.fit(X_train,Y_train)
    Y_pred = model_skf.predict(X_test)
    print( 'Predicted output in this fold ... ' , Y_pred)
    print( 'But actual output in this fold was ... ' , Y_test)
    acc = accuracy_score(Y_pred , Y_test)
    print( 'The accuracy found for (SKF k-fold) in this fold is : ',acc) 
print('ENDED')
