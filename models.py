import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from constants import output_file, y_name, path_name
from sklearn.experimental import enable_halving_search_cv  # Abilita gli estimatori di ricerca di riduzione progressiva
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV


seed = 42
data = pd.read_csv(output_file)
name = None

X = data.drop(y_name, axis=1)
y = data[y_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed, shuffle=True)
X_train = X_train.drop(path_name, axis=1)
X_test = X_test.drop(path_name, axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


'''
name = "xgb"
xgb = XGBClassifier(use_label_encoder=False, verbosity=2, seed=seed)
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.]
}
halving_grid_search = HalvingGridSearchCV(estimator=xgb, param_grid=xgb_param_grid, cv=3, factor=2, verbose=2)
halving_grid_search.fit(X_train, y_train)
print("Best XGB Accuracy:", halving_grid_search.best_score_)
print("Best XGB Parameters:", halving_grid_search.best_params_)
best_model = halving_grid_search.best_estimator_
'''

'''
name = "rf"
rf = RandomForestClassifier(verbose=1, random_state=seed)
rf_param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 5, 10],
}
halving_grid_search = HalvingGridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, factor=2, verbose=2)
halving_grid_search.fit(X_train, y_train)
print("Best RF Accuracy:", halving_grid_search.best_score_)
print("Best RF Parameters:", halving_grid_search.best_params_)
best_model = halving_grid_search.best_estimator_
'''

'''
name = "knn"
knn = KNeighborsClassifier()

knn_param_grid = {
    'n_neighbors': [5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 3]
}
halving_grid_search = HalvingGridSearchCV(estimator=knn, param_grid=knn_param_grid, cv=3, factor=2, verbose=2)
halving_grid_search.fit(X_train, y_train)
print("Best KNN Accuracy:", halving_grid_search.best_score_)
print("Best KNN Parameters:", halving_grid_search.best_params_)
best_model = halving_grid_search.best_estimator_
'''

name = "ada"
ada = AdaBoostClassifier(random_state=seed, algorithm="SAMME")
ada_param_grid = {
    'learning_rate': [0.5, 1, 1.5, 2],
    'n_estimators': [50, 100, 150, 200]
}
halving_grid_search = HalvingGridSearchCV(estimator=ada, param_grid=ada_param_grid, cv=3, factor=2, verbose=2)
halving_grid_search.fit(X_train, y_train)
print("Best ADA Accuracy:", halving_grid_search.best_score_)
print("Best ADA Parameters:", halving_grid_search.best_params_)
best_model = halving_grid_search.best_estimator_

# ensemble = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf), ('knn', knn)], voting='hard')

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
fn = 'confusion_matrix.png' if name is None else f"confusion_matrix_{name}.png"
plt.savefig(fn)
plt.show()
