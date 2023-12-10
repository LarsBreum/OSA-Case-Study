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

#SVC
print("--------- SVC ---------- ")
df_OSA = pd.read_excel("OSA_extreme_male.xlsx")

columns = df_OSA.columns.tolist()
columns = [c for c in columns if c not in ["IAH", "Patient", "OSA", "Gender"]]

#Defining OSA = severe as 1 and OSA = healthy as 0
df_OSA["Class"] = np.where((df_OSA["OSA"] == "Severe"), 1, 0)

X = np.array(df_OSA[columns])
y = np.array(df_OSA['Class'])

fig, ax = subplots(figsize=(8,8))

ax.scatter(X[:,0],
        X[:,1],
        c=y,
        cmap=cm.coolwarm);

(X_train, X_test, y_train, y_test) = skm.train_test_split(X, y, test_size=0.5, random_state=0)

svm = SVC(kernel="rbf", gamma=0.01, C=1)
svm.fit(X_train , y_train)

fig, ax = subplots(figsize=(8 ,8))
plot_svm(X_train,
        y_train,
        svm,
        ax=ax)

print("--- Predictions ---")
y_hat_test = svm.predict(X_test)
print("Predict: ",y_hat_test)

print("CONFUSION TABLE")
print(confusion_table(y_hat_test , y_test))

plt.show()