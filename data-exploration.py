import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS , summarize , poly)
from scipy import stats

OSA_df = pd.read_excel("Info_BDApnea_QuironMalaga.xlsx")

#print("-------HEAD-------")
#print(OSA_df.head())
#print("------DESCRIBE------")
print(OSA_df.describe())

#print("-----COUNT NaN------")
#print(df.loc[:, OSA_df.columns].count())

df_clean = pd.read_excel("OSA_DB_UPM_Clean.xlsx")

# Result = df.groupby('Gender')['IAH'].mean()
# print(Result)

#print(df_clean)

df_clean['bmi'] = df_clean.Weight / (df_clean.Height/100)**2
#print(df_clean)

bins = np.linspace(0, 100, 50)
plt.hist(df_clean['IAH'], bins, label='IAH')

# multiple linear regression not taking gender into account
x = MS(['Age', 'Cervical', 'bmi']).fit_transform(df_clean)
y = df_clean['IAH']

model = sm.OLS(y, x)
results = model.fit()
print("--- ML for both no interaction ---")
print(summarize(results))
print("R2: " + str(results.rsquared))

df_women = df_clean.drop(df_clean[df_clean.Gender == 1].index)
df_men = df_clean.drop(df_clean[df_clean.Gender == 0].index)


# multiple linear regression only for women
x = MS(['Age', 'Cervical', 'bmi']).fit_transform(df_women)
y = df_women['IAH']

model_1 = sm.OLS(y, x)
results_1 = model_1.fit()

print("---ML for Women---")
print(summarize(results_1))
print("R2: " + str(results_1.rsquared))


# multiple linear regression with interaction only for men

x = MS(['Age', 'Cervical', 'bmi']).fit_transform(df_men)
y = df_men['IAH']
model_2 = sm.OLS(y, x)
results_2 = model_2.fit()

print("---ML for Men ---")
print(summarize(results_2))
print("R2: " + str(results_2.rsquared))



# multiple linear regression with interaction for both

x = MS(['Age', 'bmi', ('Gender', 'Cervical')]).fit_transform(df_clean)
y = df_clean['IAH']
model = sm.OLS(y, x)
results = model.fit()

print("---ML for both w. interaction---")
print(summarize(results))
print("R2: " + str(results.rsquared))

#matrix_women = df_women.iloc[:, 1:].corr().round(2)
#sns.heatmap(matrix_women, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')

matrix_men = df_men.iloc[:, 1:].corr().round(2)
sns.heatmap(matrix_men, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')

plt.show()



# interaction for both genders


# print("----VARIANCE WOMEN----")
# print(df_women['IAH'].var())
# print("----VARIANCE MEN----")
# print(df_men['IAH'].var())

# print("All describe:")
# print(df_clean.describe())
# print("Women describe:")
# print(df_women.describe())
# print("Men describe:")
# print(df_men.describe())


# df_women_severe = df_women.drop(df_women[df_women.IAH < 30].index)
# hist = df_women_severe.plot.hist(column="bmi", range=[15, 50])
# hist.set_title("Histogram w. IAH >= 30")

# df_women_healthy = df_women.drop(df_women[df_women.IAH > 10].index)

# hist = df_women_healthy.plot.hist(column="bmi", range=[15, 50])
# hist.set_title("Histogram w. IAH < 10")

df_men_severe = df_men.drop(df_men[df_men.IAH < 30].index)

print(df_men_severe.describe())

bins = np.linspace(20, 55, 25)
plt.hist(df_men_severe['bmi'], bins, label='bmi')
plt.hist(df_men_severe.IAH, bins, label='AIH')

#Kruskal-Wallis test
bmi_IAH_kruskal = stats.kruskal(df_men_severe['bmi'], df_men_severe.IAH)
print("bmi: " + str(bmi_IAH_kruskal))

cervical_IAH_kruskal = stats.kruskal(df_men_severe['Cervical'], df_men_severe.IAH)
print("Cervical " + str(cervical_IAH_kruskal))

age_IAH_kruskal = stats.kruskal(df_men_severe['Age'], df_men_severe.IAH)
print("Age: " + str(age_IAH_kruskal))

weight_IAH_kruskal = stats.kruskal(df_men_severe['Weight'], df_men_severe.IAH)
print("Weight: " + str(weight_IAH_kruskal))

height_IAH_kruskal = stats.kruskal(df_men_severe['Height'], df_men_severe.IAH)
print("Height: " + str(height_IAH_kruskal))

all_IAH_kruskal = stats.kruskal(df_men_severe['bmi'],
                                df_men_severe['Cervical'],
                                df_men_severe['Age'],
                                df_men_severe['Weight'], 
                                df_men_severe['Height'], 
                                df_men_severe.IAH)

print("All: " + str(all_IAH_kruskal))

# plt.hist(df_men_severe['Cervical'], bins, label='cervical')
# plt.hist(df_men_severe.IAH, bins, label='AIH')

# plt.hist(df_men_severe['Age'], bins, label='Age')
# plt.hist(df_men_severe.IAH, bins, label='AIH')

#hist.set_title("Histogram w. IAH >= 30, cervical")
# print(df_men_severe.describe())
df_men_healthy = df_men.drop(df_men[df_men.IAH > 10].index)
#hist = df_men_healthy.plot.hist(column="Cervical", range=[30, 55], bins=20)
# hist.set_title("Histogram w. IAH < 10, cervical")
# print(df_men_healthy.describe())
sns.pairplot(df_men, kind="reg")

print(df_men.describe())

plt.show()