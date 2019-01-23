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

# grouping factor 1: p7 + g14


# grouping factor 2: g2 + g4 + g6


# grouping factor 3: p2
# to_exclude = info_panss.loc[info_panss["P2"]==3].index
to_exclude = info_panss.loc[((info_panss["P2"]==3) | (info_panss["P2"]==4))].index
to_exclude.tolist()
y_3 = info_panss.drop(to_exclude, axis=0).P2[205:345].copy()
y_3[y_3 < 3] = 0
# y_3[y_3 > 3] = 1
y_3[y_3 > 4] = 1

# # grouping factor 4: n5
# # to_exclude = info_panss.loc[info_panss["N5"]==3].index
# to_exclude = info_panss.loc[((info_panss["N5"]==3) | (info_panss["N5"]==4))].index
# to_exclude.tolist()
# y_4 = info_panss.drop(to_exclude, axis=0).N5[205:345].copy()
# y_4[y_4 < 3] = 0
# # y_4[y_4 > 3] = 1
# y_4[y_4 > 4] = 1

# # grouping factor 5: g11
# # to_exclude = info_panss.loc[info_panss["G11"]==3].index
# to_exclude = info_panss.loc[((info_panss["G11"]==3) | (info_panss["G11"]==4))].index
# to_exclude.tolist()
# y_5 = info_panss.drop(to_exclude, axis=0).G11[205:345].copy()
# y_5[y_5 < 3] = 0
# # y_5[y_4 > 3] = 1
# y_5[y_5 > 4] = 1

# # grouping factor 6: p3
# to_exclude = info_panss.loc[info_panss["P3"]==3].index
to_exclude = info_panss.loc[((info_panss["P3"]==3) | (info_panss["P3"]==4))].index
to_exclude.tolist()
y_6 = info_panss.drop(to_exclude, axis=0).P3[205:345].copy()
# y_6[y_6 < 3] = 0
# y_6[y_6 > 3] = 1
# y_6[y_6 > 4] = 1


# Choose the dataset to use
FILENAME1 = 'T1_246'
# X1 = pd.read_csv(DATASETS_DIR + FILENAME1 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1) # NC_SZ
# X1 = pd.read_csv(DATASETS_DIR + FILENAME1 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345] # FE_CH
X1 = pd.read_csv(DATASETS_DIR + FILENAME1 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345].drop(to_exclude, axis=0) # PANSS
FILENAME2 = 'fMRI_246'
# X2 = pd.read_csv(DATASETS_DIR + FILENAME2 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1) # NC_SZ
# X2 = pd.read_csv(DATASETS_DIR + FILENAME2 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345] # FE_CH
X2 = pd.read_csv(DATASETS_DIR + FILENAME2 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345].drop(to_exclude, axis=0) # PANSS
# columns_X2 = X2.columns.tolist()
# X2 = X2.drop(columns_X2[0:246], axis=1)
# print(X2.columns)
FILENAME3 = 'DTI_246'
# X3 = pd.read_csv(DATASETS_DIR + FILENAME3 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1) # NC_SZ
# X3 = pd.read_csv(DATASETS_DIR + FILENAME3 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345] # FE_CH
X3 = pd.read_csv(DATASETS_DIR + FILENAME3 + '.csv', encoding='gbk').drop(['ID', 'GROUP'], axis=1).iloc[205:345].drop(to_exclude, axis=0) # PANSS
FILENAME = 'COMB_246'
X = pd.concat([X1, X2, X3], axis=1)