#kfold types1
import os
import numpy as np
os.system('cls')
print('starting**')
x = np.arange(1,51).reshape(10,5)
y = np.array([0,1,1,0,1,0,0,1,0,0])#,1,0,]) 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)

from sklearn.model_selection import KFold
k_fol_spl=6
kf = KFold(n_splits=k_fol_spl,shuffle=False)
print('k-fold n_splits used :',k_fol_spl)
print('Output of - k fold for n_split \n')
for i, (train_index, test_index) in enumerate(kf.split(x)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")


##############
print('doing Kfold once again')
x = np.arange(1,25).reshape(8,3)
y = np.array([0,1,1,0,1,0,0,1,])#0,0])|
from sklearn.model_selection import KFold
k_fol_spl = 5
kf = KFold(n_splits = k_fol_spl, shuffle = True)
print('k-fold n_splits used :', k_fol_spl)

for i, (train_index, test_index) in enumerate(kf.split(x)):
    print(f"Fold {i}:")
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index] , y[test_index]
    print( 'Total Test samples = (Total samples / k-fold) \n' , 5/k_fol_spl)
    print(f"Test Values   =\n",x_test)
    print(f"Train Values  = \n",x_train)

#####chatgpt code corrected
import numpy as np
from sklearn.model_selection import KFold

x = np.arange(1, 25).reshape(8, 3)
y = np.array([0, 1, 1, 0, 1, 0, 0, 1])  # Removed extra comma

k_fol_spl = 5
kf = KFold(n_splits=k_fol_spl, shuffle=True)
print('k-fold n_splits used:', k_fol_spl)

for i, (train_index, test_index) in enumerate(kf.split(x)):
    print(f"Fold {i}:")
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(f"Total Test samples = {len(test_index)}")  # Corrected total test samples calculation
    print(f"Test Values   =\n{x_test}")
    print(f"Train Values  =\n{x_train}")

#kfold with logistic regressions
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


x,y =make_classification(n_samples=20, n_features=9, n_informative=5, n_redundant=2, random_state=1)
print('Input X is \n',x[0:4])
print('The corresponding output u is \n',y[0:4])
k_fol_spl = 3
kf = KFold(n_splits=k_fol_spl, shuffle=True)
print('k-fold n_splits used:', k_fol_spl)

for i, (train_index, test_index) in enumerate(kf.split(x)):
    print(f"Fold {i}:")
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(f"Total Test samples = {len(test_index)}")  # Corrected total test samples calculation
    print(f"Test Values   =\n{x_test}")
    print(f"Train Values  =\n{x_train}")
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print('pred output in this fold..',y_pred)
print( 'But actual output in this fold was ... ' , y_test)
acc = accuracy_score(y_pred , y_test)
acc*100
