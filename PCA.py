import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

file = 'OSA_extreme_male.xlsx'
xl = pd.ExcelFile(file)
df_OSA = pd.read_excel("OSA_extreme_male.xlsx")

df = df_OSA[['Weight','Height','Age','Cervical','BMI']]

pca = PCA(n_components = 5)

scaler = StandardScaler()
scaler.fit(df)

X_scaled = scaler.transform(df)

pca.fit(X_scaled)
PC1 = pca.components_[0]
PC2 = pca.components_[1]

features = df.columns

# plt.figure(figsize=(10,5))
# plt.subplot(121)
# plt.barh(features,PC1)
# plt.title("PC1")
# plt.subplot(122)
# plt.title("PC2")
# plt.barh(features,PC2)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');

## project data into PC space

# Z1 = pca.transform(X_scaled)[:,0] # see 'prcomp(my_data)$x' in R
# Z2 = pca.transform(X_scaled)[:,1]

# group = df_OSA['OSA']
# cdict = {'Severe': 'red', 'Healthy': 'blue'}


# fig, ax = plt.subplots()
# for g in np.unique(group):
#   ix = np.where(group == g)
#   ax.scatter(Z1[ix], Z2[ix], c = cdict[g], label = g, s = 50)

# plt.xlabel("PC1 - Obesity related",fontsize=14)
# plt.ylabel("PC2 - Height/Age related",fontsize=14)
# ax.legend()

# print(df.describe())

# plt.show()


x_new = pca.transform(X_scaled)

color= ['red' if l == 'Severe' else 'blue' for l in df_OSA['OSA']]

def myplot(score,coeff,n_var,labels=None):
    # n_var number of variables to show in biplot
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = color)
    
    plot_scale = 1.2
    
    for i in range(n_var):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 1)
        if labels is None:
            plt.text(coeff[i,0]* plot_scale, coeff[i,1] * plot_scale, features[i], color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* plot_scale, coeff[i,1] * plot_scale, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-0.8,0.8)
    plt.ylim(-0.8,0.8)
    plt.xlabel("PC{}".format(1) + " Obesity related")
    plt.ylabel("PC{}".format(2) + " Height/Age related")
    plt.grid()

#Call the function. Use only the 2 PCs.
plt.figure(figsize=(10,8))
myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]),5)
plt.show()