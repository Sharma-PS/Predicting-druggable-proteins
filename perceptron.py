import creatingFeatures as cf
import re
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys

try :
    fileNames = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
except:
    print("Arguments missing. Four arguments should be there.")
    exit()

featureNames = "AAC,DPC,CTD,PAAC,APAAC,RSacid,RSpolar,RSsecond,RScharge,RSDHP".split(',')
labels = ['Negative', 'Positive']
RANDOM_SEED = 42

# Load scikit's classifier libraries
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import sklearn.metrics as skmet

from sklearn.preprocessing import MinMaxScaler


def read_fasta(file):
    line1 = open("./data/" + file + ".txt").read().split('>')[1:]
    line2 = [item.split('\n')[0:-1] for item in line1]
    fasta = [[item[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', ''.join(item[1:]).upper())] for item in line2]
    return fasta


def createFeatureVectors(fasta, sign):
    feat = cf.Features()

    feat_AAC = feat.AAC(fasta)[0]
    feat_DPC = feat.DPC(fasta, 0)[0]
    feat_CTD = np.hstack((feat.CTDC(fasta)[0], feat.CTDD(fasta)[0], feat.CTDT(fasta)[0]))
    feat_PAAC = feat.PAAC(fasta, 1)[0]
    feat_APAAC = feat.APAAC(fasta, 1)[0]
    feat_RSacid = feat.reducedACID(fasta)
    feat_RSpolar = feat.reducedPOLAR(fasta)
    feat_RSsecond = feat.reducedSECOND(fasta)
    feat_RScharge = feat.reducedCHARGE(fasta)
    feat_RSDHP = feat.reducedDHP(fasta)

    feat_list = [feat_AAC,
                 feat_DPC,
                 feat_CTD,
                 feat_PAAC,
                 feat_APAAC,
                 feat_RSacid,
                 feat_RSpolar,
                 feat_RSsecond,
                 feat_RScharge,
                 feat_RSDHP]

    # print([len(item[0]) for item in feat_list])

    df_main = pd.DataFrame()
    for i, item in tqdm(enumerate(feat_list)):
        df = pd.DataFrame(item, columns=[f"{featureNames[i]}_{id}" for id in range(1, len(item[0]) + 1)])
        df_main = pd.concat([df_main, df], axis=1)

    df_main["TARGET"] = 1 if sign == "+" else 0

    return df_main


def createDataset(dataframes):
    df_final = pd.DataFrame()
    for df in dataframes:
        df_final = pd.concat([df_final, df], axis=0, ignore_index=True)

    return df_final


def scaleAllFeatures(feature_vecs):
    scaler = MinMaxScaler()
    scaler.fit_transform(feature_vecs)

    return scaler

def initializeModels():
    ET_clf = ExtraTreesClassifier(random_state=RANDOM_SEED)
    SVM_clf = SVC(random_state=RANDOM_SEED, probability=True)
    RF_clf = RandomForestClassifier(random_state=RANDOM_SEED)
    LR_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

    return [RF_clf, SVM_clf, LR_reg, ET_clf]


def cv(clf, X, y, nr_fold):
    ix = []
    for i in range(0, len(y)):
        ix.append(i)
    ix = np.array(ix)

    allACC = []
    allSENS = []
    allSPEC = []
    allMCC = []
    allAUC = []
    for j in range(0, nr_fold):
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf.fit(train_X, train_y)
        p = clf.predict(test_X)
        pr = clf.predict_proba(test_X)[:, 1]
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        test_y = np.array(test_y)
        for i in range(0, len(test_y)):
            if test_y[i] == 0 and p[i] == 0:
                TP += 1
            elif test_y[i] == 0 and p[i] == 1:
                FN += 1
            elif test_y[i] == 1 and p[i] == 0:
                FP += 1
            elif test_y[i] == 1 and p[i] == 1:
                TN += 1
        ACC = (TP + TN) / (TP + FP + TN + FN)
        SENS = TP / (TP + FN)
        SPEC = TN / (TN + FP)
        det = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        if (det == 0):
            MCC = 0
        else:
            MCC = ((TP * TN) - (FP * FN)) / det
        AUC = skmet.roc_auc_score(test_y, pr)
        allACC.append(ACC)
        allSENS.append(SENS)
        allSPEC.append(SPEC)
        allMCC.append(MCC)
        allAUC.append(AUC)

    return np.mean(allACC), np.mean(allSENS), np.mean(allSPEC), np.mean(allMCC), np.mean(allAUC)


def trainWithSigleFeature(df, feature):
    train_X = df.drop("TARGET", axis=1)[[col for col in df.columns if col.startswith(feature)]]
    train_y = df["TARGET"]

    RF_clf, SVM_clf, LR_reg, ET_clf = initializeModels()

    RF_clf.fit(train_X, train_y)
    SVM_clf.fit(train_X, train_y)
    LR_reg.fit(train_X, train_y)
    ET_clf.fit(train_X, train_y)

    return RF_clf, SVM_clf, LR_reg, ET_clf


def testWithSingleFeature(clf, df, feature):
    test_X = df.drop("TARGET", axis=1)[[col for col in df.columns if col.startswith(feature)]]
    test_y = df["TARGET"]

    p = clf.predict(test_X)
    pr = clf.predict_proba(test_X)[:, 1]
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(0, len(test_y)):
        if test_y[i] == 0 and p[i] == 0:
            TP += 1
        elif test_y[i] == 0 and p[i] == 1:
            FN += 1
        elif test_y[i] == 1 and p[i] == 0:
            FP += 1
        elif test_y[i] == 1 and p[i] == 1:
            TN += 1
    ACC = (TP + TN) / (TP + FP + TN + FN)
    SENS = TP / (TP + FN)
    SPEC = TN / (TN + FP)
    det = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if (det == 0):
        MCC = 0
    else:
        MCC = ((TP * TN) - (FP * FN)) / det

    AUC = skmet.roc_auc_score(test_y, pr)
    F1 = skmet.f1_score(test_y, p)
    PRECISION = skmet.precision_score(test_y, p)

    return ACC, SENS, SPEC, MCC, AUC, F1, PRECISION

# Read the Each FASTA Files
fs_tr_neg, fs_tr_pos, fs_ts_neg, fs_ts_pos = [read_fasta(file) for file in fileNames]

df_train = createDataset([createFeatureVectors(fs_tr_neg, "-"), createFeatureVectors(fs_tr_pos, "+")])
df_test = createDataset([createFeatureVectors(fs_ts_neg, "-"), createFeatureVectors(fs_ts_pos, "+")])

# Scale the values in all Features
scaler = scaleAllFeatures(df_train)
df_train_scaled = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns)
df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

# Cross Validate the Each Models
cv_scores = []
for feature in tqdm(featureNames):
    clfs = initializeModels()
    X = df_train_scaled.drop("TARGET", axis=1)[[col for col in df_train_scaled.columns if col.startswith(feature)]]
    y = df_train_scaled["TARGET"]
    for i, clf_str in tqdm(enumerate(["RF", "SVC", "LR", "ET"])):
        score = list(cv(clfs[i], X, y, 5))
        score.append(f"{clf_str}_{feature}")
        cv_scores.append(score)

df_cv_score = pd.DataFrame(cv_scores, columns=["ACC", "SENS", "SPEC", "MCC", "AUC", "MODEL-FEATURE"])

df_final_cv_scores = pd.DataFrame()
for feature in featureNames:
    df = df_cv_score[[val.endswith(f'_{feature}') for val in df_cv_score["MODEL-FEATURE"].values]].sort_values(["MCC"], ascending=False).head(1)
    df_final_cv_scores = pd.concat([df_final_cv_scores, df], axis=0, ignore_index=True)

# Hyperparameter tuning
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import joblib

kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

def SVC_objective(trial:Trial, X_train, y_train):    
    C = trial.suggest_int('C', 1, 10)
    kernel = trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf'])
    degree =trial.suggest_int("degree", 1, 5)
    gamma = trial.suggest_categorical("gamma", ["scale","auto"])
    max_iter = trial.suggest_int("max_iter", -1, 100)

    svc= SVC(
        kernel=kernel,
        C=C, degree = degree,
        gamma= gamma,
        probability = True,
        max_iter=max_iter,
        random_state=RANDOM_SEED
    )

    scores = cross_val_score(svc, X_train, y_train, cv=kfolds, scoring="accuracy")

    return scores.mean()

def RF_objective(trial:Trial, X_train, y_train):
    _n_estimators = trial.suggest_int("n_estimators", 25, 250)
    _max_depth = trial.suggest_int("max_depth", 5, 20)
    _criterion = trial.suggest_categorical("criterion",["gini", "entropy", "log_loss"])
    _min_samp_split = trial.suggest_int("min_samples_split", 2, 10)
    _min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
    _max_features = trial.suggest_int("max_features", 10, 50)
    _ccp_alpha = trial.suggest_float("ccp_alpha",0, 0.05)


    rf = RandomForestClassifier(
        max_depth=_max_depth,
        min_samples_split=_min_samp_split,
        min_samples_leaf=_min_samples_leaf,
        n_estimators=_n_estimators,
        n_jobs=-1,
        ccp_alpha=_ccp_alpha,
        criterion=_criterion,
        max_features=_max_features,
        random_state=RANDOM_SEED
    )

    scores = cross_val_score(rf, X_train, y_train, cv=kfolds, scoring="accuracy")

    return scores.mean()

def ET_objective(trial:Trial, X_train, y_train):
    _n_estimators = trial.suggest_int("n_estimators", 25, 250)
    _max_depth = trial.suggest_int("max_depth", 5, 20)
    _criterion = trial.suggest_categorical("criterion",["gini", "entropy", "log_loss"])
    _min_samp_split = trial.suggest_int("min_samples_split", 2, 10)
    _min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
    _max_features = trial.suggest_int("max_features", 10, 50)
    _ccp_alpha = trial.suggest_float("ccp_alpha", 0, 0.05)

    et = ExtraTreesClassifier(
        max_depth=_max_depth,
        min_samples_split=_min_samp_split,
        min_samples_leaf=_min_samples_leaf,
        n_estimators=_n_estimators,
        n_jobs=-1,
        ccp_alpha=_ccp_alpha,
        criterion=_criterion,
        max_features=_max_features,
        random_state=RANDOM_SEED
    )

    scores = cross_val_score(et, X_train, y_train, cv=kfolds, scoring="accuracy")

    return scores.mean()

def LR_objective(trial:Trial, X_train, y_train):
    C = trial.suggest_float("C", 1.0, 10.0, log=True)
    tol = trial.suggest_float("tol", 0.0001, 0.01, log=True)
    max_iter = trial.suggest_int("max_iter", 1000, 6000, 200)

    lr = LogisticRegression(
         C=C,
         tol=tol,
         multi_class="auto",
         max_iter=max_iter,
         random_state=RANDOM_SEED,
    )

    scores = cross_val_score(lr, X_train, y_train, cv=kfolds, scoring="accuracy")

    return scores.mean()

def tune(objective):

    # create a seed for the sampler for reproducibility
    sampler = TPESampler(seed=RANDOM_SEED) 
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=100)

    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}\n")
    print(f"Optimized parameters: {params}\n")
    
    return params

# Train the Each Models with default parameters
test_scores_0 = []

for model_feature in tqdm(df_final_cv_scores["MODEL-FEATURE"].values):
    clf, feature = model_feature.split('_')

    X_train = df_train_scaled.drop("TARGET", axis=1)[[col for col in df_train_scaled.columns if col.startswith(feature)]]
    y_train = df_train_scaled["TARGET"]

    RF_clf, SVM_clf, LR_reg, ET_clf = initializeModels()

    if(clf == "RF"):
        model = RF_clf
    elif(clf == "SVC"):
        model = SVM_clf
    elif(clf == "LR"):
        model = LR_reg
    else:
        model = ET_clf

    model.fit(X_train, y_train)

    score = list(testWithSingleFeature(model, df_test_scaled, feature))
    score.append(model_feature)
    test_scores_0.append(score)

df_test_score_0 = pd.DataFrame(test_scores_0, columns=["ACC", "SENS", "SPEC", "MCC", "AUC", "F1", "PRECISION", "MODEL_FEATURE"])

print(df_test_score_0)

# ============================================================================================================================

# Get best params the Each Models
# ht_cv_scores = {}
# for model_feature in tqdm(df_final_cv_scores["MODEL-FEATURE"].values):
#     clf, feature = model_feature.split('_')

#     X_train = df_train_scaled.drop("TARGET", axis=1)[[col for col in df_train_scaled.columns if col.startswith(feature)]]
#     y_train = df_train_scaled["TARGET"]
    
#     if(clf == "RF"):
#         # Wrap the objective inside a lambda and call objective inside it
#         func = lambda trial: RF_objective(trial, X_train, y_train)
#     elif(clf == "SVC"):
#         func = lambda trial: SVC_objective(trial, X_train, y_train)
#     elif(clf == "LR"):
#         func = lambda trial: LR_objective(trial, X_train, y_train)
#     else:
#         func = lambda trial: ET_objective(trial, X_train, y_train)

#     params = tune(func)
#     ht_cv_scores[f"{clf}_{feature}"] = params

# ================================================================================================================================

# Train the Each Models with optimized parameters
test_scores_1 = []
base_leaners = joblib.load('./models/baseleaners.h5')

for model_feature, model in tqdm(base_leaners):
    _, feature = model_feature.split('_')

    X_train = df_train_scaled.drop("TARGET", axis=1)[[col for col in df_train_scaled.columns if col.startswith(feature)]]
    y_train = df_train_scaled["TARGET"]

    score = list(testWithSingleFeature(model, df_test_scaled, feature))
    score.append(model_feature)
    test_scores_1.append(score)

df_test_score_1 = pd.DataFrame(test_scores_1, columns=["ACC", "SENS", "SPEC", "MCC", "AUC", "F1", "PRECISION", "MODEL_FEATURE"])

print(df_test_score_1)

def get_all_predictions(X: pd.DataFrame):

    pr = []

    for model_feature, leaner in base_leaners:
        _, feature = model_feature.split('_')

        p = leaner.predict_proba(X.loc[:, [col for col in X.columns if col.startswith(feature)]])[:, 0]
        pr.append(p)

    pr1, pr2, pr3, pr4, pr5, pr6, pr7, pr8, pr9, pr10 = pr

    allpr = np.hstack((pr1.reshape((len(pr1), 1)), pr2.reshape((len(pr1), 1)), pr3.reshape((len(pr1), 1)),
                   pr4.reshape((len(pr1), 1)), pr5.reshape((len(pr1), 1)), pr6.reshape((len(pr1), 1)),
                   pr7.reshape((len(pr1), 1)), pr8.reshape((len(pr1), 1)), pr9.reshape((len(pr1), 1)),
                   pr10.reshape((len(pr1), 1))))
    
    return allpr    

X_train = df_train_scaled.drop("TARGET", axis=1)
y_train = df_train_scaled["TARGET"]

train_allPr = get_all_predictions(X_train)

scaler_final = MinMaxScaler()

train_allPr_scl = scaler_final.fit_transform(train_allPr)

meta_leaner = joblib.load('./models/metaleaners.h5')

X_test = df_test_scaled.drop("TARGET", axis=1)
y_test = df_test_scaled["TARGET"]

test_allPr = get_all_predictions(X_test)

test_allPr_scl = scaler_final.transform(test_allPr)

p_label = meta_leaner.predict(test_allPr_scl)

import matplotlib.pyplot as plt

tn, fp, fn, tp = skmet.confusion_matrix(y_true=y_test.values, y_pred=p_label).ravel()

test_final_scores = [[
    skmet.accuracy_score(y_true=y_test, y_pred=p_label),
    (tp /  (tp+fn)),
    (tn / (tn+fp)),
    skmet.matthews_corrcoef(y_true=y_test, y_pred=p_label),
    skmet.roc_auc_score(y_test, p_label),
    skmet.f1_score(y_true=y_test, y_pred=p_label),
    skmet.precision_score(y_true=y_test, y_pred=p_label)
]]

df_test_final_score = pd.DataFrame(test_final_scores, columns=["ACC", "SENS", "SPEC", "MCC", "AUC", "F1", "PRECISION"])
print(df_test_final_score)

pos_test = df_test_scaled[df_test_scaled["TARGET"] == 1.0]
neg_test = df_test_scaled[df_test_scaled["TARGET"] == 0.0]

test_pos_allPr_scl = scaler_final.transform(get_all_predictions(pos_test))
test_neg_allPr_scl = scaler_final.transform(get_all_predictions(neg_test))

pos_p_label = meta_leaner.predict(test_pos_allPr_scl)
pos_p_prob = meta_leaner.predict_proba(test_pos_allPr_scl)[:, 1]
neg_p_label = meta_leaner.predict(test_neg_allPr_scl)
neg_p_prob = meta_leaner.predict_proba(test_neg_allPr_scl)[:, 0]

file = open("./output/TS_pos_result.csv", "w")
for i, (head, seq) in enumerate(fs_ts_pos):
    file.write(head + "," + str(int(pos_p_label[i])) + "," + labels[int(pos_p_label[i])] + "," + str(pos_p_prob[i]) + "\n")
file.close()

file = open("./output/TS_neg_result.csv", "w")
for i, (head, seq) in enumerate(fs_ts_neg):
    file.write(head + "," + str(int(neg_p_label[i])) + "," + labels[int(neg_p_label[i])] + "," + str(neg_p_prob[i]) + "\n")
file.close()
