import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from constants import output_file, y_name, path_name
from sklearn.experimental import enable_halving_search_cv  # Abilita gli estimatori di ricerca di riduzione progressiva
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.base import TransformerMixin
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.optimizers import Adam, RMSprop

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


def remove_outliers(df, q1=0.25, q3=0.75):
    for col in df.columns:
        if col != y_name:
            Q1 = df[col].quantile(q1)
            Q3 = df[col].quantile(q3)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def plot_confusion_matrix(y_test, y_pred, name=None):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    fn = 'confusion_matrix.png' if name is None else f"confusion_matrix_{name}.png"
    plt.savefig(fn)
    # plt.show()


def print_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    print(f'Accuracy: {round(accuracy, 2)}')
    print(f'Precision: {round(precision, 2)}')
    print(f'Recall: {round(recall, 2)}')


seed = 42
data = pd.read_csv(output_file)
data = data.drop(
    columns=["mfcc_mean_1", "mfcc_mean_4", "mfcc_mean_6", "gfcc_mean_2", "mfcc_mean_8", "mfcc_mean_7", "mfcc_mean_10",
             path_name])
# data = remove_outliers(data)
name = None

X = data.drop(y_name, axis=1)
clipper = ColumnWiseOutlierClipper(lower_percentile=2.5, upper_percentile=97.5)
X = clipper.fit_transform(X)
y = data[y_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed, shuffle=True)

xgb_model = False
rf_model = False
knn_model = False
ada_model = False
svc_model = False
nn_model = False
ensemble = True

if xgb_model:
    scaler = PowerTransformer()
    X_train_xgb = scaler.fit_transform(X_train)
    X_test_xgb = scaler.transform(X_test)

    name = "xgb"
    xgb = XGBClassifier(use_label_encoder=False, verbosity=2, seed=seed)
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9, 1.]
    }
    halving_grid_search = HalvingGridSearchCV(estimator=xgb, param_grid=xgb_param_grid, cv=3, factor=2, verbose=2)
    halving_grid_search.fit(X_train_xgb, y_train)
    print("Best XGB Accuracy:", round(halving_grid_search.best_score_, 2))
    print("Best XGB Parameters:", halving_grid_search.best_params_)
    best_model = halving_grid_search.best_estimator_
    y_pred_xgb = best_model.predict(X_test_xgb)
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_xgb, name=name)
    print_metrics(y_test=y_test, y_pred=y_pred_xgb)

if rf_model:
    scaler = PowerTransformer()
    X_train_rf = scaler.fit_transform(X_train)
    X_test_rf = scaler.transform(X_test)

    name = "rf"
    rf = RandomForestClassifier(verbose=1, random_state=seed)
    rf_param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 5, 10],
    }
    halving_grid_search = HalvingGridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, factor=2, verbose=2)
    halving_grid_search.fit(X_train_rf, y_train)
    print("Best RF Accuracy:", round(halving_grid_search.best_score_, 2))
    print("Best RF Parameters:", halving_grid_search.best_params_)
    best_model = halving_grid_search.best_estimator_
    y_pred_rf = best_model.predict(X_test_rf)
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_rf, name=name)
    print_metrics(y_test=y_test, y_pred=y_pred_rf)


if knn_model:
    scaler = MinMaxScaler()
    X_train_knn = scaler.fit_transform(X_train)
    X_test_knn = scaler.transform(X_test)
    name = "knn"
    knn = KNeighborsClassifier()

    knn_param_grid = {
        'n_neighbors': [5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3]
    }
    halving_grid_search = HalvingGridSearchCV(estimator=knn, param_grid=knn_param_grid, cv=3, factor=2, verbose=2)
    halving_grid_search.fit(X_train_knn, y_train)
    print("Best KNN Accuracy:", round(halving_grid_search.best_score_, 2))
    print("Best KNN Parameters:", halving_grid_search.best_params_)
    best_model = halving_grid_search.best_estimator_
    y_pred_knn = best_model.predict(X_test_knn)
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_knn, name=name)
    print_metrics(y_test=y_test, y_pred=y_pred_knn)

if ada_model:
    scaler = PowerTransformer()
    X_train_ada = scaler.fit_transform(X_train)
    X_test_ada = scaler.transform(X_test)
    name = "ada"
    ada = AdaBoostClassifier(random_state=seed, algorithm="SAMME")
    ada_param_grid = {
        'learning_rate': [0.5, 1, 1.5, 2],
        'n_estimators': [50, 100, 150, 200]
    }
    halving_grid_search = HalvingGridSearchCV(estimator=ada, param_grid=ada_param_grid, cv=3, factor=2, verbose=2)
    halving_grid_search.fit(X_train_ada, y_train)
    print("Best ADA Accuracy:", round(halving_grid_search.best_score_, 2))
    print("Best ADA Parameters:", halving_grid_search.best_params_)
    best_model = halving_grid_search.best_estimator_
    y_pred_ada = best_model.predict(X_test_ada)
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_ada, name=name)
    print_metrics(y_test=y_test, y_pred=y_pred_ada)

if svc_model:
    scaler = MinMaxScaler()
    X_train_svc = scaler.fit_transform(X_train)
    X_test_svc = scaler.transform(X_test)
    name = "svc"
    svc = SVC(random_state=seed)
    svc_param_grid = {
        'C': [0.5, 1, 2],
        'kernel': ['rbf', 'sigmoid']
    }
    halving_grid_search = HalvingGridSearchCV(estimator=svc, param_grid=svc_param_grid, cv=3, factor=2, verbose=2)
    halving_grid_search.fit(X_train_svc, y_train)
    print("Best SVC Accuracy:", round(halving_grid_search.best_score_, 2))
    print("Best SVC Parameters:", halving_grid_search.best_params_)
    best_model = halving_grid_search.best_estimator_
    y_pred_svc = best_model.predict(X_test_svc)
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_svc, name=name)
    print_metrics(y_test=y_test, y_pred=y_pred_svc)

if nn_model:
    tf.random.set_seed(seed)
    scaler = MinMaxScaler()
    X_train_nn = scaler.fit_transform(X_train)
    X_test_nn = scaler.transform(X_test)

    name = "nn"
    input_layer = layers.Input(shape=(X_train.shape[1],))
    x = layers.Dense(256)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        verbose=1,
        restore_best_weights=True,
    )
    history = model.fit(X_train_nn, y_train, epochs=50, batch_size=512, validation_split=0.1, callbacks=[early_stopping])
    predictions = model.predict(X_test_nn)
    y_pred_nn = (predictions > 0.5).astype("int32")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_nn, name=name)
    print_metrics(y_test=y_test, y_pred=y_pred_nn)
    # ensemble = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf), ('knn', knn)], voting='hard')

if ensemble:
    pw_scaler = PowerTransformer()
    mm_scaler = MinMaxScaler()
    X_train_pw = pw_scaler.fit_transform(X_train)
    X_test_pw = pw_scaler.transform(X_test)
    X_train_mm = mm_scaler.fit_transform(X_train)
    X_test_mm = mm_scaler.transform(X_test)
    xgb = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=300,
                        subsample=0.8, verbosity=2, seed=seed)
    rf = RandomForestClassifier(verbose=1, random_state=seed, max_depth=None, n_estimators=400)
    knn = KNeighborsClassifier(n_neighbors=9, p=1, weights='distance')
    name = "ensemble"
    models = [xgb, rf, knn]
    xgb.fit(X_train_pw, y_train)
    rf.fit(X_train_pw, y_train)
    knn.fit(X_train_mm, y_train)

    pred_xgb = xgb.predict(X_test_pw)
    pred_rf = rf.predict(X_test_pw)
    pred_knn = knn.predict(X_test_mm)

    final_predictions = []
    for i in range(len(pred_xgb)):
        count_1 = (pred_xgb[i] == 1) + (pred_rf[i] == 1) + (pred_knn[i] == 1)
        count_0 = (pred_xgb[i] == 0) + (pred_rf[i] == 0) + (pred_knn[i] == 0)
        if count_1 >= count_0:
            final_predictions.append(1)
        else:
            final_predictions.append(0)
    final_predictions = np.array(final_predictions)
    plot_confusion_matrix(y_test=y_test, y_pred=final_predictions, name=name)
    print_metrics(y_test=y_test, y_pred=final_predictions)
