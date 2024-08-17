import pandas as pd
from constants import output_file, y_name, path_name, seed
from matplotlib import pyplot as plt
import seaborn as sns, numpy as np
from sklearn.base import TransformerMixin
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# df = pd.read_csv(output_file)
# features = list(df.columns)
# for c in [y_name, path_name]:
#     features.remove(c)
# X = df[features]
# y = df[y_name]
# class_0 = df[df[y_name] == 0]
# class_1 = df[df[y_name] == 1]
# print(f"Class 0: {class_0.shape}, Class 1: {class_1.shape}")
# df.drop(columns=[y_name, path_name], inplace=True)


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


def compute_importance():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = XGBClassifier(use_label_encoder=False, verbosity=2, seed=seed)
    model.fit(X_train, y_train)

    print("Computing Feature Importance")
    importance = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.barh(importance_df['Feature'], importance_df['Importance'], color="b", align="center")
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png")

    print("Computing Permutation Importance")
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=-1)
    importances = result.importances_mean
    std = result.importances_std
    indices = np.argsort(importances)[::-1]
    perm_df = pd.DataFrame({'Feature': X.columns[indices], 'Importance': importances[indices]})
    plt.figure(figsize=(12, 8))
    plt.title("Permutation Importance delle Feature")
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig("permutation_importance.png")
    return importance_df, perm_df

# importance_df, permutation_df = compute_importance()
# neg_importance = permutation_df[permutation_df['Importance'] < 0]
# print(neg_importance)

# print(df.var())
# print_correlation()
# print_outliers(df)
# print_kde()
# print(df.shape)
# new_df = remove_outliers(q1=0.1, q3=0.9)
# print(new_df.shape)
# print_outliers(df)
# clipper = ColumnWiseOutlierClipper(lower_percentile=2.5, upper_percentile=97.5)
# df_clipped = clipper.fit_transform(df)
# print_outliers(df_clipped, "box_cox_clipped.png")


import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

'''

# Supponendo che il file CSV si chiami 'data.csv'
df = pd.read_csv('test_result.csv')

# Estrai le colonne rilevanti
y_true = df['true_label']
y_pred = df['predicted_label']

# Calcolo dell'accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calcolo della precisione
precision = precision_score(y_true, y_pred)

# Calcolo del recall
recall = recall_score(y_true, y_pred)

# Stampa i risultati
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
'''

df = pd.read_csv("test_result.csv")
paths = df["path"].values
leg = []
mal = []

for path in paths:
    path_list = path.split("/")
    class_ = path_list[1]
    file_ = path_list[2]
    if class_ == "0":
        leg.append(file_)
    else:
        mal.append(file_)
print("Leg", len(leg))
print("Mal", len(mal))



