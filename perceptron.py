# Imports
import creatingFeatures as cf
import pandas as pd
import re
import numpy as np
import math

fileNames = ["TR_neg_SPIDER", "TR_pos_SPIDER", "TS_neg_SPIDER", "TS_pos_SPIDER"]
featureNames = "AAC,DPC,CTD,PAAC,APAAC,RSacid,RSpolar,RSsecond,RScharge,RSDHP".split(',')
labels = ['Positive', 'Negative']


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

    for i, item in enumerate(feat_list):
        df = pd.DataFrame(item, columns=[f"{featureNames[i]}_{id}" for id in range(1, len(item[0]) + 1)])
        df_main = pd.concat([df_main, df], axis=1)

    df_main["TARGET"] = 1 if sign == "+" else 0

    return df_main


def createDataset(dataframes):
    df_final = pd.DataFrame()

    for df in dataframes:
        # df_tmp = df[[col for col in df.columns if col.startswith(feature)]]
        df_final = pd.concat([df_final, df], axis=0, ignore_index=True)

    return df_final


def scaleAllFeatures(feature_vecs):
    scaler = MinMaxScaler()

    scaler.fit_transform(feature_vecs)

    return scaler


# Load scikit's classifier libraries
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support

from sklearn.preprocessing import MinMaxScaler


def initializeModels():
    # Create a random forest Classifier. By convention, clf means 'Classifier'
    RF_clf = RandomForestClassifier(n_jobs=2, random_state=0)

    # Create a SVM Classifier
    SVM_clf = SVC(random_state=0, C=10, probability=True)

    # Create a Logistic Regression Classifier
    LR_reg = LogisticRegression(solver='lbfgs', max_iter=1000)

    # Create a ExtraTree Classifier
    ET_clf = ExtraTreesClassifier(random_state=0)

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
        AUC = roc_auc_score(test_y, pr)
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
    AUC = roc_auc_score(test_y, pr)

    print(precision_recall_fscore_support(y_true=test_y, y_pred=p))

    return ACC, SENS, SPEC, MCC, AUC


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
for feature in featureNames:
    clfs = initializeModels()
    X = df_train_scaled.drop("TARGET", axis=1)[[col for col in df_train_scaled.columns if col.startswith(feature)]]
    y = df_train_scaled["TARGET"]
    for i, clf_str in enumerate(["RF", "SVC", "LR", "ET"]):
        score = list(cv(clfs[i], X, y, 5))
        score.append(f"{clf_str}_{feature}")
        cv_scores.append(score)

df_cv_score = pd.DataFrame(cv_scores, columns=["ACC", "SENS", "SPEC", "MCC", "AUC", "MODEL-FEATURE"])

print(df_cv_score)

# Train the Models for Each Features
clfs = trainWithSigleFeature(df_train_scaled, "AAC")
print([testWithSingleFeature(clf, df_test_scaled, "AAC") for clf in clfs])
