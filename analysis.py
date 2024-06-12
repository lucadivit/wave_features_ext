import pandas as pd
from constants import output_file, y_name, path_name
from matplotlib import pyplot as plt
import seaborn as sns, numpy as np
from sklearn.base import TransformerMixin

df = pd.read_csv(output_file)
df.drop(columns=[y_name, path_name], inplace=True)


class ColumnWiseOutlierClipper(TransformerMixin):
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        self.lower_bounds_ = X.apply(lambda col: np.percentile(col, self.lower_percentile), axis=0)
        self.upper_bounds_ = X.apply(lambda col: np.percentile(col, self.upper_percentile), axis=0)
        return self

    def transform(self, X):
        X_clipped = X.copy()
        for col in X.columns:
            X_clipped[col] = np.clip(X_clipped[col], self.lower_bounds_[col], self.upper_bounds_[col])
        return X_clipped


def print_correlation():
    '''
    Dalla matrice di correlazione possono essere rimosse queste colonne:
    mfcc_mean_4, mfcc_mean_6, gfcc_mean_2, mfcc_mean_8, mfcc_mean_7, mfcc_mean_10
    :return:
    '''
    # Spearman per non essere soggetti agli outlier
    corr_matrix = df.corr(method='spearman').round(2)
    plt.figure(figsize=(25, 25))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matrice di Correlazione')
    plt.savefig("Correlation.png")


def print_outliers(df, fn=None):
    '''
    Dal grafico si vede che ci sono molti outliers. Potrebbe valer la pena trasformarli
    '''
    plt.figure(figsize=(30, 8))
    sns.boxplot(data=df)
    plt.title('Boxplot delle variabili')
    if fn is None:
        plt.savefig("BoxPlot.png")
    else:
        plt.savefig(fn)


def print_kde():
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['mfcc_mean_0'], shade=True)
    plt.title('KDE Plot mfcc_mean_0')
    plt.xlabel('Valori')
    plt.ylabel('Densità')
    plt.show()


def remove_outliers(q1=0.25, q3=0.75):
    '''
    Dallo studio degli outlier la colonna mfcc_mean_1 può essere levata in quanto molto distorta
    '''
    df = pd.read_csv(output_file)
    df.drop(columns=[y_name, path_name], inplace=True)
    for col in df.columns:
        Q1 = df[col].quantile(q1)
        Q3 = df[col].quantile(q3)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


# print_correlation()
# print_outliers()
# print_kde()
# print(df.shape)
# new_df = remove_outliers(q1=0.1, q3=0.9)
# print(new_df.shape)
print_outliers(df)
clipper = ColumnWiseOutlierClipper(lower_percentile=2.5, upper_percentile=97.5)
df_clipped = clipper.fit_transform(df)
print_outliers(df_clipped, "box_cox_clipped.png")
