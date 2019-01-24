import pandas as pd

WORKING_DIR = '~/Projects/neuroimaging_with_python/'
DATASETS_DIR = WORKING_DIR + 'datasets/181228/'

info = pd.read_csv(DATASETS_DIR + 'info.csv', encoding='gbk')
y_all = info.GROUP[0:345].copy() # NC vs SZ
y_all[266:345] = 1
y_sz = info.GROUP[205:345].copy() # FE vs CH
y_sz[y_sz == 1] = 0
y_sz[y_sz == 2] = 1
y_p = info.P_ALL[205:345].copy() # P_ALL
y_n = info.N_ALL[205:345].copy() # N_ALL
y_g = info.G_ALL[205:345].copy() # G_ALL
y_s = info.SCOREP1[205:345].copy() # SCOREP1

info_panss = pd.read_csv(DATASETS_DIR + '../190117/345_info_panss.csv', encoding='gbk')

class PANSS_INDEX():
    def __init__(self, labelname, action):
        self.name = labelname
        self.action = action

    def select(self):
        to_select = info_panss.loc[205:345].index

        if self.name == 'P7+G14':
            if self.action == 'CLF':
                to_select = info_panss.loc[(((info_panss['P7'] < 3) & (info_panss['G14'] < 3)) | ((info_panss['P7'] > 4) & (info_panss['G14'] > 4)))].index
                to_select.tolist()
                y_panss = info_panss.P7[to_select].copy() + info_panss.G14[to_select].copy()
                y_panss[y_panss < 5] = 0
                # y_panss[y_panss > 7] = 1
                y_panss[y_panss > 9] = 1
            elif self.action == 'RGS':
                y_panss = info_panss.P7[to_select].copy() + info_panss.G14[to_select].copy()

        elif self.name == 'G2+G4+G6':
            if self.action == 'CLF':
                to_select = info_panss.loc[(((info_panss['G2'] < 3) & (info_panss['G4'] < 3) & (info_panss['G6'] < 3)) | ((info_panss['G2'] > 4) & (info_panss['G4'] > 4) & (info_panss['G6'] > 4)))].index
                to_select.tolist()
                y_panss = info_panss.G2[to_select].copy() + info_panss.G4[to_select].copy() + info_panss.G6[to_select].copy()
                y_panss[y_panss < 7] = 0
                # y_panss[y_panss > 10] = 1
                y_panss[y_panss > 13] = 1
            elif self.action == 'RGS':
                y_panss = info_panss.G2[to_select].copy() + info_panss.G4[to_select].copy() + info_panss.G6[to_select].copy()

        elif self.name == 'P2':
            if self.action == 'CLF':
                to_select = info_panss.loc[((info_panss['P2'] < 3) | (info_panss['P2'] > 4))].index
                to_select.tolist()
                y_panss = info_panss.P2[to_select].copy()
                y_panss[y_panss < 3] = 0
                # y_panss[y_panss > 3] = 1
                y_panss[y_panss > 4] = 1
            elif self.action == 'RGS':
                y_panss = info_panss.P2[to_select].copy()

        elif self.name == 'N5':
            if self.action == 'CLF':
                to_select = info_panss.loc[((info_panss['N5'] < 3) | (info_panss['N5'] > 4))].index
                to_select.tolist()
                y_panss = info_panss.N5[to_select].copy()
                y_panss[y_panss < 3] = 0
                # y_panss[y_panss > 3] = 1
                y_panss[y_panss > 4] = 1
            elif self.action == 'RGS':
                y_panss = info_panss.N5[to_select].copy()

        elif self.name == 'G11':
            if self.action == 'CLF':
                to_select = info_panss.loc[((info_panss['G11'] < 3) | (info_panss['G11'] > 4))].index
                to_select.tolist()
                y_panss = info_panss.G11[to_select].copy()
                y_panss[y_panss < 3] = 0
                # y_panss[y_panss > 3] = 1
                y_panss[y_panss > 4] = 1
            elif self.action == 'RGS':
                y_panss = info_panss.G11[to_select].copy()

        elif self.name == 'P3':
            if self.action == 'CLF':
                to_select = info_panss.loc[((info_panss['P3'] < 3) | (info_panss['P3'] > 4))].index
                to_select.tolist()
                y_panss = info_panss.P3[to_select].copy()
                y_panss[y_panss < 3] = 0
                # y_panss[y_panss > 3] = 1
                y_panss[y_panss > 4] = 1
            elif self.action == 'RGS':
                y_panss = info_panss.P3[to_select].copy()

        return y_panss, to_select

"""
:labelname: P7+G14; G2+G4+G6; P2; N5; G11; P3.
:action: CLF; RGS
"""
panss_index = PANSS_INDEX('P3', 'CLF') # change the index and action here
y_panss, to_select = panss_index.select()

"""
Choose the dataset to use.
T1: T1_90; T1_246. GMV; WMV.
fMRI: fMRI_90; fMRI_246. ALFF; DC; ReHo.
DTI: DTI_90; DTI_246. FA; L1; L23M; MD.
"""
T1 = 'T1_246'
# X_T1 = pd.read_csv(DATASETS_DIR + T1 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1) # NC_SZ
# X_T1 = pd.read_csv(DATASETS_DIR + T1 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345] # FE_CH
X_T1 = pd.read_csv(DATASETS_DIR + T1 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).loc[to_select] # PANSS
fMRI = 'fMRI_246'
# X_fMRI = pd.read_csv(DATASETS_DIR + fMRI + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1) # NC_SZ
# X_fMRI = pd.read_csv(DATASETS_DIR + fMRI + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345] # FE_CH
X_fMRI = pd.read_csv(DATASETS_DIR + fMRI + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).loc[to_select] # PANSS
DTI = 'DTI_246'
# X_DTI = pd.read_csv(DATASETS_DIR + DTI + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1) # NC_SZ
# X_DTI = pd.read_csv(DATASETS_DIR + DTI + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345] # FE_CH
X_DTI = pd.read_csv(DATASETS_DIR + DTI + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).loc[to_select] # PANSS
COMB = 'COMB_246'
X_COMB = pd.concat([X_T1, X_fMRI, X_DTI], axis=1)