import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")

# Check data integrity

df = pd.read_csv("wdat.csv")
print(df.info())
print(df.describe())
print(df.columns)
print(df.head(5))

# Preparing and making sure conversion is successfull

df["Datum"] = pd.to_datetime(df["Datum"])
df["Tid"] = pd.to_datetime(df["Tid"])
df = df.dropna(axis=1)
print(df.tail())
print(df.dtypes)

# Because time is important, we concatenate date and time

df["Datumtid"] = pd.to_datetime(df["Datum"], ' ',  df["Tid"])
df = df.drop(["Datum", "Tid"], axis=1)
df = df.set_index("Datumtid")

sns.heatmap(df.corr(), annot=True)
plt.title("Correlation of all columns in the dataframe")
plt.show()

# Fetch the monthly mean values and perform analysis

df_column = ["Lufttemperatur", "Daggpunktstemperatur"]
df_monthly_mean = df[df_column].resample("MS").mean()
print(df_monthly_mean.head())

sns.set_style("darkgrid")
sns.regplot(data=df_monthly_mean, x="Lufttemperatur", y="Daggpunktstemperatur", color="b")
plt.title("Relation between temperature (C) and dew-point")
plt.show()

sns.lineplot(data=df_monthly_mean)
plt.xlabel("Year")
plt.title("Variation of temperature and dew-point over time")
plt.show()

sns.set_style("darkgrid")
sns.FacetGrid(df, hue="Nederbördsmängd", height=10).map(plt.scatter, "Lufttemperatur", "Daggpunktstemperatur").add_legend()
plt.show()

# Time to start preparing our data for predictions!

def variance_results(y_train, y_test):
    variance = metrics.explained_variance_score(y_train, y_test)
    mean_absolute_error = metrics.mean_absolute_error(y_train, y_test)
    mse = metrics.mean_squared_error(y_train, y_test)
    msle = metrics.mean_squared_log_error(y_train, y_test)
    r2 = metrics.r2_score(y_train, y_test)

    print("Variance: ", round(variance, 3))
    print("Mean squared log error: ", round(msle, 3))
    print("R2: ", round(r2, 3))
    print("MAE: ", round(mean_absolute_error, 3))
    print("MSE: ", round(mse, 3))
    print("RMSE: ", round(np.sqrt(mse), 4))

# We create a new df containing data from our air temperature column

df_precip = df
df_precip.loc[:, "Yesterday"] = df_precip.loc[:, "Lufttemperatur"].shift()
df_precip.loc[:, "Yesterday_diff"] = df_precip.loc[:, "Yesterday"].diff()
df_precip = df_precip.dropna()

# Preparing X and Y

X_train = df_precip[:"2020"].drop(["Lufttemperatur"], axis=1)
y_train = df_precip.loc[:"2020", "Lufttemperatur"]

X_test = df_precip[:"2021"].drop(["Lufttemperatur"], axis=1)
y_test = df_precip.loc[:"2021", "Lufttemperatur"]

models = []
models.append(("KNN", KNeighborsRegressor()))
models.append(("RF", RandomForestRegressor(n_estimators=10)))
models.append(("MLP", MLPRegressor()))

results = []
names = []
for name, model in models:
    tscv = TimeSeriesSplit(n_splits=10)

    cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring="r2")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

plt.boxplot(results, labels=names)
plt.title("Comparison")
plt.show()
