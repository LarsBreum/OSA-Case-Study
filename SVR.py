import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots, cm
import matplotlib.pyplot as plt
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table

from sklearn.svm import SVC
from sklearn.svm import SVR
from ISLP.svm import plot as plot_svm
from sklearn.metrics import RocCurveDisplay

#SVR
print(" --------- SVR --------- ")
df_OSA = pd.read_excel("OSA_Regression.xlsx")

df = df_OSA[['Weight','Height','Age','Cervical','BMI']]

### Picking predictor columns

columns = df_OSA.columns.tolist()

# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c not in ["IAH"]]
# Store the variable we'll be predicting on.
target = "IAH"

print('Predictors: ',columns)


X = np.array(df_OSA[columns])
y = np.array(df_OSA['IAH'])

fig, ax = subplots(figsize=(8,8))

ax.scatter(X[:,0],
        X[:,1],
        c=y,
        cmap=cm.coolwarm);

(X_train, 
 X_test, 
 y_train, 
 y_test) = skm.train_test_split(X, 
                                y, 
                                test_size=0.3, 
                                random_state=0)

svm = SVR(kernel="rbf", gamma=0.01, C=1)
svm.fit(X_train , y_train)

fig, ax = subplots(figsize=(8 ,8))
plot_svm(X_train,
        y_train,
        svm,
        ax=ax)

y_hat_test = svm.predict(X_test)

plt.show()