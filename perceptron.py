# Imports
import creatingFeatures as cf
import pandas as pd
import re
import numpy as np

fileNames = ["TR_neg_SPIDER", "TR_pos_SPIDER", "TS_neg_SPIDER", "TS_pos_SPIDER"]


def read_fasta(file):
    line1 = open("./data/" + file + ".txt").read().split('>')[1:]
    line2 = [item.split('\n')[0:-1] for item in line1]
    fasta = [[item[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', ''.join(item[1:]).upper())] for item in line2]
    return fasta

def createDataset(fasta, datatype):
    featNames = "AAC,DPC,CTD,PAAC,APAAC,RSacid,RSpolar,RSsecond,RScharge,RSDHP".split(',')
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

    print(
        len(feat_AAC[0]),
        len(feat_APAAC[0]),
        len(feat_CTD[0]),
        len(feat_DPC[0]),
        len(feat_PAAC[0]),
        len(feat_RSacid[0]),
        len(feat_RScharge[0]),
        len(feat_RSsecond[0]),
        len(feat_RSpolar[0]),
        len(feat_RSDHP[0]), )
    df_main = pd.DataFrame()
    for i, item in enumerate([feat_AAC,feat_DPC, feat_CTD, feat_PAAC, feat_APAAC, feat_RSacid, feat_RSpolar, feat_RSsecond, feat_RScharge, feat_RSDHP]):
        df = pd.DataFrame(item, columns= [f"{featNames[i]}_{id}" for id in range(1, len(item[0])+1)])
        df_main = pd.concat([df_main, df], axis = 1)

    print(df_main)



    df = pd.DataFrame()

fs = read_fasta(fileNames[1])
createDataset(fs,"TR")