import pandas as pd, random, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from constants import (output_file, y_name, path_name, seed, n_ceps, output_test_fn, output_train_fn,
                       summary_fn, output_validation_fn, threshold_fn, train_set_fn, test_set_fn, validation_set_fn)
from sklearn.experimental import enable_halving_search_cv  # Abilita gli estimatori di ricerca di riduzione progressiva
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.base import TransformerMixin
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.optimizers import Adam, RMSprop
import pickle


def save_model(path: str, model):
    with open(f'{path}.pkl', 'wb') as file:
        pickle.dump(model, file)

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


def print_metrics(y_pred, y_test, type):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    print(f"----------{type.upper()}-----------")
    print(f'Accuracy: {round(accuracy, 2)}')
    print(f'Precision: {round(precision, 2)}')
    print(f'Recall: {round(recall, 2)}')
    print(f"------------------------------------")
    return accuracy, precision, recall


def get_validation_set(df: pd.DataFrame):
    raw_df = df.copy()
    df = df.copy()
    paths = df[path_name].values
    service = [path.split("/")[2] for path in paths]
    df["service"] = service
    sub_df_leg = df[df[y_name] == 0]
    sub_df_neg = df[df[y_name] == 1]
    leg_services = list(set(sub_df_leg["service"].values))
    mal_services = list(set(sub_df_neg["service"].values))
    random.seed(seed)
    leg_services_random = random.choices(leg_services, k=10)
    mal_services_random = random.choices(mal_services, k=10)
    sub_df_leg_random = df[df['service'].isin(leg_services_random)]
    sub_df_mal_random = df[df['service'].isin(mal_services_random)]
    index_to_rm = list(sub_df_leg_random.index) + list(sub_df_mal_random.index)
    raw_df.drop(index_to_rm, inplace=True)
    validation_set = pd.concat([sub_df_leg_random, sub_df_mal_random], ignore_index=True)
    validation_set.drop(columns="service", inplace=True)
    return validation_set, raw_df


def save_df(X, y, name):
    df = X.copy()
    df[y_name] = y
    df.to_csv(f"{name}", index=False)


name = None
load = True
clipper = ColumnWiseOutlierClipper(lower_percentile=2.5, upper_percentile=97.5)

if load:
    print("Loading Old Files")
    X_train = pd.read_csv(train_set_fn)
    y_train = X_train[y_name]
    X_train = X_train.drop(y_name, axis=1)
    train_path = X_train[path_name]
    X_train = X_train.drop(path_name, axis=1)
    X_train = clipper.fit_transform(X_train)

    X_test = pd.read_csv(test_set_fn)
    y_test = X_test[y_name]
    X_test = X_test.drop(y_name, axis=1)
    test_path = X_test[path_name]
    X_test = X_test.drop(path_name, axis=1)
    X_test = clipper.transform(X_test)

    X_val = pd.read_csv(validation_set_fn)
    y_val = X_val[y_name]
    X_val = X_val.drop(y_name, axis=1)
    validation_path = X_val[path_name]
    X_val = X_val.drop(path_name, axis=1)
    X_val = clipper.transform(X_val)
else:
    data = pd.read_csv(output_file)
    data = data.drop(
        columns=["mfcc_mean_0", "mfcc_mean_2", "mfcc_mean_4", "mfcc_mean_5",
                 "mfcc_mean_6", "mfcc_mean_8", "mfcc_mean_10", "bfcc_mean_4",
                 "bfcc_mean_12"])

    validation_set, data = get_validation_set(df=data)
    X = data.drop(y_name, axis=1)
    y = data[y_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed, shuffle=True)
    save_df(X=X_train, y=y_train, name=train_set_fn)
    save_df(X=X_test, y=y_test, name=test_set_fn)
    train_path = X_train[path_name]
    test_path = X_test[path_name]
    X_train = X_train.drop(path_name, axis=1)
    X_train = clipper.fit_transform(X_train)
    X_test = X_test.drop(path_name, axis=1)
    X_test = clipper.transform(X_test)

    y_val = validation_set[y_name]
    X_val = validation_set.drop(y_name, axis=1)
    save_df(X=X_val, y=y_val, name=validation_set_fn)
    validation_path = X_val[path_name]
    X_val = X_val.drop(path_name, axis=1)
    X_val = clipper.transform(X_val)

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
        'n_estimators': [300, 400, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.6, 0.8, 1.]
    }
    halving_grid_search = HalvingGridSearchCV(estimator=xgb, param_grid=xgb_param_grid, cv=3, factor=2, verbose=2)
    halving_grid_search.fit(X_train_xgb, y_train)
    print("Best XGB Accuracy:", round(halving_grid_search.best_score_, 2))
    print("Best XGB Parameters:", halving_grid_search.best_params_)
    best_model = halving_grid_search.best_estimator_
    y_pred_xgb = best_model.predict(X_test_xgb)
    y_pred_xgb_train = best_model.predict(X_train_xgb)
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_xgb, name=name)
    print_metrics(y_test=y_train, y_pred=y_pred_xgb_train, type="train")
    print_metrics(y_test=y_test, y_pred=y_pred_xgb, type="test")

if rf_model:
    scaler = PowerTransformer()
    X_train_rf = scaler.fit_transform(X_train)
    X_test_rf = scaler.transform(X_test)

    name = "rf"
    rf = RandomForestClassifier(verbose=1, random_state=seed)
    rf_param_grid = {
        'n_estimators': [300, 400, 500],
        'max_depth': [None, 4, 6],
    }
    halving_grid_search = HalvingGridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, factor=2, verbose=2)
    halving_grid_search.fit(X_train_rf, y_train)
    print("Best RF Accuracy:", round(halving_grid_search.best_score_, 2))
    print("Best RF Parameters:", halving_grid_search.best_params_)
    best_model = halving_grid_search.best_estimator_
    y_pred_rf = best_model.predict(X_test_rf)
    y_pred_rf_train = best_model.predict(X_train_rf)
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_rf, name=name)
    print_metrics(y_test=y_train, y_pred=y_pred_rf_train, type="train")
    print_metrics(y_test=y_test, y_pred=y_pred_rf, type="test")

if knn_model:
    scaler = MinMaxScaler()
    X_train_knn = scaler.fit_transform(X_train)
    X_test_knn = scaler.transform(X_test)
    name = "knn"
    knn = KNeighborsClassifier()

    knn_param_grid = {
        'n_neighbors': [3, 4, 5],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3]
    }
    halving_grid_search = HalvingGridSearchCV(estimator=knn, param_grid=knn_param_grid, cv=3, factor=2, verbose=2)
    halving_grid_search.fit(X_train_knn, y_train)
    print("Best KNN Accuracy:", round(halving_grid_search.best_score_, 2))
    print("Best KNN Parameters:", halving_grid_search.best_params_)
    best_model = halving_grid_search.best_estimator_
    y_pred_knn = best_model.predict(X_test_knn)
    y_pred_knn_train = best_model.predict(X_train_knn)
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_knn, name=name)
    print_metrics(y_test=y_train, y_pred=y_pred_knn_train, type="train")
    print_metrics(y_test=y_test, y_pred=y_pred_knn, type="test")

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
    y_pred_ada_train = best_model.predict(X_train_ada)
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_ada, name=name)
    print_metrics(y_test=y_train, y_pred=y_pred_ada_train, type="train")
    print_metrics(y_test=y_test, y_pred=y_pred_ada, type="test")

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
    y_pred_svc_train = best_model.predict(X_train_svc)
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_svc, name=name)
    print_metrics(y_test=y_train, y_pred=y_pred_svc_train, type="train")
    print_metrics(y_test=y_test, y_pred=y_pred_svc, type="test")


def create_nn():
    tf.random.set_seed(seed)
    input_layer = layers.Input(shape=(X_train.shape[1],))
    x = layers.Dense(256)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=RMSprop(learning_rate=0.0002),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_nn(model):
    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        verbose=1,
        mode="max",
        restore_best_weights=True,
    )
    history = model.fit(X_train_nn, y_train, epochs=100, batch_size=512, validation_split=0.2,
                        callbacks=[early_stopping])
    return history


if nn_model:
    scaler = MinMaxScaler()
    X_train_nn = scaler.fit_transform(X_train)
    X_test_nn = scaler.transform(X_test)

    name = "nn"

    model = create_nn()
    history = train_nn(model)
    predictions = model.predict(X_test_nn)
    y_pred_nn = (predictions > 0.5).astype("int32")
    y_pred_nn_train = (model.predict(X_train_nn) > 0.5).astype("int32")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    plot_confusion_matrix(y_test=y_test, y_pred=y_pred_nn, name=name)
    print_metrics(y_test=y_train, y_pred=y_pred_nn_train, type="train")
    print_metrics(y_test=y_test, y_pred=y_pred_nn, type="test")
    # ensemble = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf), ('knn', knn)], voting='hard')

if ensemble:

    def compute_threshold_preds(y_test, predictions, optimal_threshold=None):
        if optimal_threshold is None:
            fpr, tpr, thresholds = roc_curve(y_test, predictions)
            gmeans = np.sqrt(tpr * (1 - fpr))
            ix = np.argmax(gmeans)
            optimal_threshold = thresholds[ix]
            pred = (predictions >= optimal_threshold).astype(int)
        else:
            pred = (predictions >= optimal_threshold).astype(int)
        return pred, optimal_threshold


    pw_scaler = PowerTransformer()
    mm_scaler = MinMaxScaler()
    X_train_pw = pw_scaler.fit_transform(X_train)
    X_test_pw = pw_scaler.transform(X_test)
    X_val_pw = pw_scaler.transform(X_val)
    X_train_mm = mm_scaler.fit_transform(X_train)
    X_test_mm = mm_scaler.transform(X_test)
    X_val_mm = mm_scaler.transform(X_val)

    xgb = XGBClassifier(learning_rate=0.2, max_depth=6, n_estimators=300,
                        subsample=0.8, verbosity=2, random_state=seed)
    rf = RandomForestClassifier(verbose=1, random_state=seed, max_depth=None, n_estimators=300)
    knn = KNeighborsClassifier(n_neighbors=4, p=1, weights='distance')
    name = "ensemble"
    models = [xgb, rf, knn]
    xgb.fit(X_train_pw, y_train)
    rf.fit(X_train_pw, y_train)
    knn.fit(X_train_mm, y_train)

    save_model(path="XGB", model=xgb)
    save_model(path="RF", model=rf)
    save_model(path="KNN", model=knn)
    save_model(path="PowerTransform", model=pw_scaler)
    save_model(path="MinMaxTransform", model=mm_scaler)
    save_model(path="Clipper", model=clipper)

    # pred_xgb = xgb.predict(X_test_pw)
    # pred_rf = rf.predict(X_test_pw)
    # pred_knn = knn.predict(X_test_mm)
    #
    pred_xgb_train = xgb.predict(X_train_pw)
    pred_rf_train = rf.predict(X_train_pw)
    pred_knn_train = knn.predict(X_train_mm)

    pred_xgb, xgb_thr = compute_threshold_preds(y_test=y_test, predictions=xgb.predict_proba(X_test_pw)[:, 1])
    pred_rf, rf_thr = compute_threshold_preds(y_test=y_test, predictions=rf.predict_proba(X_test_pw)[:, 1])
    pred_knn, knn_thr = compute_threshold_preds(y_test=y_test, predictions=knn.predict_proba(X_test_mm)[:, 1])

    pred_xgb_val, _ = compute_threshold_preds(y_test=None, predictions=xgb.predict_proba(X_val_pw)[:, 1], optimal_threshold=xgb_thr)
    pred_rf_val, _ = compute_threshold_preds(y_test=None, predictions=rf.predict_proba(X_val_pw)[:, 1], optimal_threshold=rf_thr)
    pred_knn_val, _ = compute_threshold_preds(y_test=None, predictions=knn.predict_proba(X_val_mm)[:, 1], optimal_threshold=knn_thr)

    print(f"XGB Thr: {xgb_thr}")
    print(f"RF Thr: {rf_thr}")
    print(f"KNN Thr: {knn_thr}")

    data = {"XGB_Thr": float(xgb_thr), "RF_Thr": float(rf_thr), "KNN_Thr": float(knn_thr)}
    with open(threshold_fn, 'w') as file:
        json.dump(data, file, indent=4)

    final_predictions_test = []
    for i in range(len(pred_xgb)):
        count_1 = (pred_xgb[i] == 1) + (pred_rf[i] == 1) + (pred_knn[i] == 1)
        count_0 = (pred_xgb[i] == 0) + (pred_rf[i] == 0) + (pred_knn[i] == 0)
        if count_1 >= count_0:
            final_predictions_test.append(1)
        else:
            final_predictions_test.append(0)
    final_predictions_test = np.array(final_predictions_test)

    final_predictions_train = []
    for i in range(len(pred_xgb_train)):
        count_1 = (pred_xgb_train[i] == 1) + (pred_rf_train[i] == 1) + (pred_knn_train[i] == 1)
        count_0 = (pred_xgb_train[i] == 0) + (pred_rf_train[i] == 0) + (pred_knn_train[i] == 0)
        if count_1 >= count_0:
            final_predictions_train.append(1)
        else:
            final_predictions_train.append(0)
    final_predictions_train = np.array(final_predictions_train)

    final_predictions_validation = []
    for i in range(len(pred_xgb_val)):
        count_1 = (pred_xgb_val[i] == 1) + (pred_rf_val[i] == 1) + (pred_knn_val[i] == 1)
        count_0 = (pred_xgb_val[i] == 0) + (pred_rf_val[i] == 0) + (pred_knn_val[i] == 0)
        if count_1 >= count_0:
            final_predictions_validation.append(1)
        else:
            final_predictions_validation.append(0)
    final_predictions_validation = np.array(final_predictions_validation)

    plot_confusion_matrix(y_test=y_test, y_pred=final_predictions_test, name=name)
    print_metrics(y_test=y_train, y_pred=final_predictions_train, type="train")
    print_metrics(y_test=y_test, y_pred=final_predictions_test, type="test")
    print_metrics(y_test=y_val, y_pred=final_predictions_validation, type="validation")
    pd.DataFrame({"true_label": y_train, "predicted_label": final_predictions_train, path_name: train_path}).to_csv(
        output_train_fn, index=False)
    pd.DataFrame({"true_label": y_test, "predicted_label": final_predictions_test, path_name: test_path}).to_csv(
        output_test_fn, index=False)
    pd.DataFrame({"true_label": y_val, "predicted_label": final_predictions_validation, path_name: validation_path}).to_csv(
        output_validation_fn, index=False)


def load_validation():
    final_df = pd.read_csv(output_validation_fn)
    final_df["Service"] = final_df[path_name].apply(lambda x: x.split("/")[2])
    final_df = final_df.set_index('Service')
    final_df['NumericalIndex'] = final_df.groupby(level=0).cumcount()
    final_df = final_df.set_index('NumericalIndex', append=True)
    final_df = final_df.sort_index(level=0)
    return final_df


df = load_validation()

thr_tests = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def summarize_group(group):
    true_label = list(set(group['true_label'].values))[0]
    n_rows = group.shape[0]
    nb_1s = (group['predicted_label'] == 1).sum()
    ratio = nb_1s / n_rows

    new_row = {
        'service': group.name,
        'true_label': true_label,
    }

    for perc in thr_tests:
        new_row[str(perc)] = 1 if ratio > perc else 0

    return pd.Series(new_row)

summary = df.groupby(level='Service').apply(summarize_group).reset_index(drop=True)
summary.to_csv(summary_fn, index=False)

cols = [str(elem) for elem in thr_tests]
true_values = summary["true_label"]
thr = []
acc = []
prec = []
rec = []
for col in cols:
    predicted_values = summary[col]
    accuracy, precision, recall = print_metrics(y_test=true_values, y_pred=predicted_values, type=col)
    thr.append(float(col))
    acc.append(accuracy)
    prec.append(precision)
    rec.append(recall)

pd.DataFrame({"threshold": thr, "accuracy": acc, "precision": prec, "recall": rec}).to_csv("summary_metrics.csv", index=False)
