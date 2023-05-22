import numpy as np
import math
from collections import Counter
import re


class Features:
    def __int__(self):
        self.feat_DPC = []
        self.feat_CTD = []
        self.feat_AAC = []
        self.feat_PAAC = []
        self.feat_APAAC = []
        self.feat_RSacid = []
        self.feat_RSpolar = []
        self.feat_RSsecond = []
        self.feat_RScharge = []
        self.feat_RSDHP = []

    def AAC(self, fastas, **kw):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        # AA = 'ARNDCQEGHILKMFPSTWYV'
        encodings = []
        header = []
        for i in AA:
            header.append(i)
        # encodings.append(header)

        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            count = Counter(sequence)
            for key in count:
                count[key] = count[key] / len(sequence)
            code = []
            for aa in AA:
                code.append(count[aa])
            encodings.append(code)

        return np.array(encodings, dtype=float), header

    def APAAC(self, fastas, lambdaValue=30, w=0.05, **kw):
        records = []
        records.append(
            "#	A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V")
        records.append(
            "Hydrophobicity	0.62	-2.53	-0.78	-0.9	0.29	-0.85	-0.74	0.48	-0.4	1.38	1.06	-1.5	0.64	1.19	0.12	-0.18	-0.05	0.81	0.26	1.08")
        records.append(
            "Hydrophilicity	-0.5	3	0.2	3	-1	0.2	3	0	-0.5	-1.8	-1.8	3	-1.3	-2.5	0	0.3	-0.4	-3.4	-2.3	-1.5")
        records.append(
            "SideChainMass	15	101	58	59	47	72	73	1	82	57	57	73	75	91	42	31	45	130	107	43")

        AA = ''.join(records[0].rstrip().split()[1:])
        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(1, len(records) - 1):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])

        AAProperty1 = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])

        encodings = []
        header = []
        for i in AA:
            header.append('Pc1.' + i)
        for j in range(1, lambdaValue + 1):
            for i in AAPropertyNames:
                header.append('Pc2.' + i + '.' + str(j))

        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = []
            theta = []

            for n in range(1, lambdaValue + 1):
                for j in range(len(AAProperty1)):
                    theta.append(
                        sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                             range(len(sequence) - n)]) / (len(sequence) - n))
            myDict = {}
            for aa in AA:
                myDict[aa] = sequence.count(aa)

            code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
            code = code + [w * value / (1 + w * sum(theta)) for value in theta]

            encodings.append(code)
        return np.array(encodings, dtype=float), header

    def DPC(self, fastas, gap, **kw):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
        header = [] + diPeptides

        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i

        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = []
            tmpCode = [0] * 400
            for j in range(len(sequence) - 2 + 1 - gap):
                tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + gap + 1]]] = tmpCode[AADict[sequence[j]] * 20 +
                                                                                            AADict[sequence[
                                                                                                j + gap + 1]]] + 1
            if sum(tmpCode) != 0:
                tmpCode = [i / sum(tmpCode) for i in tmpCode]
            code = code + tmpCode
            encodings.append(code)
        return np.array(encodings, dtype=float), header

    def PAAC(self, fastas, lambdaValue=30, w=0.05, **kw):
        records = []
        records.append(
            "#	A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V")
        records.append(
            "Hydrophobicity	0.62	-2.53	-0.78	-0.9	0.29	-0.85	-0.74	0.48	-0.4	1.38	1.06	-1.5	0.64	1.19	0.12	-0.18	-0.05	0.81	0.26	1.08")
        records.append(
            "Hydrophilicity	-0.5	3	0.2	3	-1	0.2	3	0	-0.5	-1.8	-1.8	3	-1.3	-2.5	0	0.3	-0.4	-3.4	-2.3	-1.5")
        records.append(
            "SideChainMass	15	101	58	59	47	72	73	1	82	57	57	73	75	91	42	31	45	130	107	43")
        AA = ''.join(records[0].rstrip().split()[1:])
        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(1, len(records)):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])

        AAProperty1 = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])

        encodings = []
        header = []
        for aa in AA:
            header.append('Xc1.' + aa)
        for n in range(1, lambdaValue + 1):
            header.append('Xc2.lambda' + str(n))

        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = []
            theta = []
            for n in range(1, lambdaValue + 1):
                theta.append(
                    sum([self.Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in
                         range(len(sequence) - n)]) / (
                            len(sequence) - n))
            myDict = {}
            for aa in AA:
                myDict[aa] = sequence.count(aa)
            code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
            code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
            encodings.append(code)
        return np.array(encodings, dtype=float), header

    def reducedACID(self, seq):
        def fcount(string, substr):
            count = 0
            pos = 0
            while (True):
                pos = string.find(substr, pos)
                if pos > -1:
                    count = count + 1
                    pos += 1
                else:
                    break
            return count

        for count, fasta in enumerate(seq):
            sub = "akn"
            subsub = [it1 + it2 for it1 in sub for it2 in sub]
            aalist = "ACDEFGHIKLMNPQRSTVWY"
            aasub = {}
            aasub["a"] = "DE"
            aasub["k"] = "KHR"
            aasub["n"] = "ACFGILMNPQSTVWY"

            seq1 = fasta[1]
            lenn = len(seq1)
            seq2 = seq1
            for key, value in aasub.items():
                for aa in value:
                    seq2 = seq2.replace(aa, key)

            freq2 = {}
            for item in sub:
                freq2[item] = fcount(seq2, item)
            for item in subsub:
                freq2[item] = fcount(seq2, item)

            freq1 = {}
            for item in aalist:
                freq1[item] = fcount(seq1, item)

            feat = []
            for key, value in aasub.items():
                feat.append(freq2[key] / lenn)

            for item in aalist:
                for key, value in aasub.items():
                    if item in value:
                        feat.append(freq1[item] / max(1, freq2[key]))
                        break

            for item in subsub:
                feat.append(freq2[item] / (freq2[item[0]] + 1))

            feat = np.array(feat)
            feat = feat.reshape(1, len(feat))
            if count == 0:
                allfeat = feat
            else:
                allfeat = np.vstack((allfeat, feat))

        return allfeat

    """
    Polarity/acidity DE RHK WYF SCMNQT GAVLIP
    Acidity DE KHR ACFGILMNPQSTVWY
    Secondary structure EHALMQKR VTIYCWF GDNPS
    Charge KR AVNCQGHILMFPSTWY DE
    DHP PALVIFWM QSTYCNG HKR DE
    """

    def reducedPOLAR(self, seq):
        def fcount(string, substr):
            count = 0
            pos = 0
            while (True):
                pos = string.find(substr, pos)
                if pos > -1:
                    count = count + 1
                    pos += 1
                else:
                    break
            return count

        for count, fasta in enumerate(seq):
            sub = "qwert"
            subsub = [it1 + it2 for it1 in sub for it2 in sub]
            aalist = "ACDEFGHIKLMNPQRSTVWY"
            aasub = {}
            aasub["q"] = "DE"
            aasub["w"] = "RHK"
            aasub["e"] = "WYF"
            aasub["r"] = "SCMNQT"
            aasub["t"] = "GAVLIP"

            seq1 = fasta[1]
            lenn = len(seq1)
            seq2 = seq1
            for key, value in aasub.items():
                for aa in value:
                    seq2 = seq2.replace(aa, key)

            freq2 = {}
            for item in sub:
                freq2[item] = fcount(seq2, item)
            for item in subsub:
                freq2[item] = fcount(seq2, item)

            freq1 = {}
            for item in aalist:
                freq1[item] = fcount(seq1, item)

            feat = []
            for key, value in aasub.items():
                feat.append(freq2[key] / lenn)

            for item in aalist:
                for key, value in aasub.items():
                    if item in value:
                        feat.append(freq1[item] / max(1, freq2[key]))
                        break

            for item in subsub:
                feat.append(freq2[item] / (freq2[item[0]] + 1))

            feat = np.array(feat)
            feat = feat.reshape(1, len(feat))
            if count == 0:
                allfeat = feat
            else:
                allfeat = np.vstack((allfeat, feat))

        return allfeat

    def reducedSECOND(self, seq):
        def fcount(string, substr):
            count = 0
            pos = 0
            while (True):
                pos = string.find(substr, pos)
                if pos > -1:
                    count = count + 1
                    pos += 1
                else:
                    break
            return count

        for count, fasta in enumerate(seq):
            sub = "qwe"
            subsub = [it1 + it2 for it1 in sub for it2 in sub]
            aalist = "ACDEFGHIKLMNPQRSTVWY"
            aasub = {}
            aasub["q"] = "EHALMQKR"
            aasub["w"] = "VTIYCWF"
            aasub["e"] = "GDNPS"

            seq1 = fasta[1]
            lenn = len(seq1)
            seq2 = seq1
            for key, value in aasub.items():
                for aa in value:
                    seq2 = seq2.replace(aa, key)

            freq2 = {}
            for item in sub:
                freq2[item] = fcount(seq2, item)
            for item in subsub:
                freq2[item] = fcount(seq2, item)

            freq1 = {}
            for item in aalist:
                freq1[item] = fcount(seq1, item)

            feat = []
            for key, value in aasub.items():
                feat.append(freq2[key] / lenn)

            for item in aalist:
                for key, value in aasub.items():
                    if item in value:
                        feat.append(freq1[item] / max(1, freq2[key]))
                        break

            for item in subsub:
                feat.append(freq2[item] / (freq2[item[0]] + 1))

            feat = np.array(feat)
            feat = feat.reshape(1, len(feat))
            if count == 0:
                allfeat = feat
            else:
                allfeat = np.vstack((allfeat, feat))

        return allfeat

    def reducedCHARGE(self, seq):
        def fcount(string, substr):
            count = 0
            pos = 0
            while (True):
                pos = string.find(substr, pos)
                if pos > -1:
                    count = count + 1
                    pos += 1
                else:
                    break
            return count

        for count, fasta in enumerate(seq):
            sub = "qwe"
            subsub = [it1 + it2 for it1 in sub for it2 in sub]
            aalist = "ACDEFGHIKLMNPQRSTVWY"
            aasub = {}
            aasub["q"] = "KR"
            aasub["w"] = "AVNCQGHILMFPSTWY"
            aasub["e"] = "DE"

            seq1 = fasta[1]
            lenn = len(seq1)
            seq2 = seq1
            for key, value in aasub.items():
                for aa in value:
                    seq2 = seq2.replace(aa, key)

            freq2 = {}
            for item in sub:
                freq2[item] = fcount(seq2, item)
            for item in subsub:
                freq2[item] = fcount(seq2, item)

            freq1 = {}
            for item in aalist:
                freq1[item] = fcount(seq1, item)

            feat = []
            for key, value in aasub.items():
                feat.append(freq2[key] / lenn)

            for item in aalist:
                for key, value in aasub.items():
                    if item in value:
                        feat.append(freq1[item] / max(1, freq2[key]))
                        break

            for item in subsub:
                feat.append(freq2[item] / (freq2[item[0]] + 1))

            feat = np.array(feat)
            feat = feat.reshape(1, len(feat))
            if count == 0:
                allfeat = feat
            else:
                allfeat = np.vstack((allfeat, feat))

        return allfeat

    def reducedDHP(self, seq):
        def fcount(string, substr):
            count = 0
            pos = 0
            while (True):
                pos = string.find(substr, pos)
                if pos > -1:
                    count = count + 1
                    pos += 1
                else:
                    break
            return count

        for count, fasta in enumerate(seq):
            sub = "qwer"
            subsub = [it1 + it2 for it1 in sub for it2 in sub]
            aalist = "ACDEFGHIKLMNPQRSTVWY"
            aasub = {}
            aasub["q"] = "PALVIFWM"
            aasub["w"] = "QSTYCNG"
            aasub["e"] = "HKR"
            aasub["r"] = "DE"

            seq1 = fasta[1]
            lenn = len(seq1)
            seq2 = seq1
            for key, value in aasub.items():
                for aa in value:
                    seq2 = seq2.replace(aa, key)

            freq2 = {}
            for item in sub:
                freq2[item] = fcount(seq2, item)
            for item in subsub:
                freq2[item] = fcount(seq2, item)

            freq1 = {}
            for item in aalist:
                freq1[item] = fcount(seq1, item)

            feat = []
            for key, value in aasub.items():
                feat.append(freq2[key] / lenn)

            for item in aalist:
                for key, value in aasub.items():
                    if item in value:
                        feat.append(freq1[item] / max(1, freq2[key]))
                        break

            for item in subsub:
                feat.append(freq2[item] / (freq2[item[0]] + 1))

            feat = np.array(feat)
            feat = feat.reshape(1, len(feat))
            if count == 0:
                allfeat = feat
            else:
                allfeat = np.vstack((allfeat, feat))

        return allfeat

    def Count(self, seq1, seq2):
        sum = 0
        for aa in seq1:
            sum = sum + seq2.count(aa)
        return sum

    def Count2(self, aaSet, sequence):
        number = 0
        for aa in sequence:
            if aa in aaSet:
                number = number + 1
        cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
        cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

        code = []
        for cutoff in cutoffNums:
            myCount = 0
            for i in range(len(sequence)):
                if sequence[i] in aaSet:
                    myCount += 1
                    if myCount == cutoff:
                        code.append((i + 1) / len(sequence) * 100)
                        break
            if myCount == 0:
                code.append(0)
        return code

    def CTDC(self, fastas, **kw):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }

        groups = [group1, group2, group3]
        property = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

        encodings = []
        header = []
        for p in property:
            for g in range(1, len(groups) + 1):
                header.append(p + '.G' + str(g))

        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = []
            for p in property:
                c1 = self.Count(group1[p], sequence) / len(sequence)
                c2 = self.Count(group2[p], sequence) / len(sequence)
                c3 = 1 - c1 - c2
                code = code + [c1, c2, c3]
            encodings.append(code)
        return np.array(encodings, dtype=float), header

    def CTDD(self, fastas, **kw):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }

        groups = [group1, group2, group3]
        property = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

        encodings = []
        header = []
        for p in property:
            for g in ('1', '2', '3'):
                for d in ['0', '25', '50', '75', '100']:
                    header.append(p + '.' + g + '.residue' + d)

        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = []
            for p in property:
                code = code + self.Count2(group1[p], sequence) + self.Count2(group2[p], sequence) + self.Count2(
                    group3[p], sequence)
            encodings.append(code)
        return np.array(encodings, dtype=float), header

    def CTDT(self, fastas, **kw):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }

        groups = [group1, group2, group3]
        property = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

        encodings = []
        header = []
        for p in property:
            for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
                header.append(p + '.' + tr)

        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            code = []
            aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
            for p in property:
                c1221, c1331, c2332 = 0, 0, 0
                for pair in aaPair:
                    if (pair[0] in group1[p] and pair[1] in group2[p]) or (
                            pair[0] in group2[p] and pair[1] in group1[p]):
                        c1221 = c1221 + 1
                        continue
                    if (pair[0] in group1[p] and pair[1] in group3[p]) or (
                            pair[0] in group3[p] and pair[1] in group1[p]):
                        c1331 = c1331 + 1
                        continue
                    if (pair[0] in group2[p] and pair[1] in group3[p]) or (
                            pair[0] in group3[p] and pair[1] in group2[p]):
                        c2332 = c2332 + 1
                code = code + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
            encodings.append(code)
        return np.array(encodings, dtype=float), header

    def Rvalue(self, aa1, aa2, AADict, Matrix):
        return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)
