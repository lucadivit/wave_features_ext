import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from constants import output_file, y_name, path_name
from sklearn.experimental import enable_halving_search_cv  # Abilita gli estimatori di ricerca di riduzione progressiva
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV


seed = 42
data = pd.read_csv(output_file)

X = data.drop(y_name, axis=1)
y = data[y_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed, shuffle=True)
X_train = X_train.drop(path_name, axis=1)
X_test = X_test.drop(path_name, axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
predictions = best_model.predict(X_test)

'''
rf = RandomForestClassifier(verbose=1, random_state=seed)
knn = KNeighborsClassifier()

param_grid = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 4, 5],
    'xgb__learning_rate': [0.05, 0.1, 0.15],
    'xgb__subsample': [0.8, 0.9, 1.],
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 5, 10],
    'knn__n_neighbors': [5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2, 3]
}

ensemble = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf), ('knn', knn)], voting='hard')
grid_search = GridSearchCV(estimator=ensemble, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy: ", grid_search.best_score_)
best_model = grid_search.best_estimator_
'''


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
plt.savefig('confusion_matrix.png')
plt.show()
