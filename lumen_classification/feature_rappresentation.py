r'''CODE TO IMPLEMENT THE GRAPHICS FEATURE SELECTION
decomment the parte of the code in order of the kind og graph that you want
Author Sonia '''

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


path = os.getcwd()
big_folder='DATA_SET'
set_folder_val='VALIDATION'
set_folder_train='TRAIN'
set_folder_test='TEST'
patch_folder='patches'
FS_folder='feature_selected'

# train load
train=np.load(os.path.join(path,big_folder,set_folder_train,'train.npy'), allow_pickle=True)
gt_train=np.load(os.path.join(path,big_folder,set_folder_train,'gt_train.npy'), allow_pickle=True)
usable_train=np.load(os.path.join(path,big_folder,set_folder_train,'usable_train.npy'), allow_pickle=True)
feature_name_train=np.load(os.path.join(path,big_folder,set_folder_train,'feature_name.npy'), allow_pickle=True)



# validation load
validation=np.load(os.path.join(path,big_folder,set_folder_val,'validation.npy'), allow_pickle=True)
gt_val=np.load(os.path.join(path,big_folder,set_folder_val,'gt_val.npy'), allow_pickle=True)
usable_val=np.load(os.path.join(path,big_folder,set_folder_val,'usable_val.npy'), allow_pickle=True)
feature_name_val=np.load(os.path.join(path,big_folder,set_folder_val,'feature_name.npy'), allow_pickle=True)

# test load
test=np.load(os.path.join(path,big_folder,set_folder_test,'test.npy'), allow_pickle=True)
gt_test=np.load(os.path.join(path,big_folder,set_folder_test,'gt_test.npy'), allow_pickle=True)
usable_test=np.load(os.path.join(path,big_folder,set_folder_test,'usable_test.npy'), allow_pickle=True)
feature_name_test=np.load(os.path.join(path,big_folder,set_folder_test,'feature_name.npy'), allow_pickle=True)

# ----------------------------------------------decomment this line to analyse just the selected feature by the automatic feature selection------------------------------------------------
# fs_KNN=np.load(os.path.join(path,FS_folder,'fs_KNN.npy'), allow_pickle=True)
# fs_RF=np.load(os.path.join(path,FS_folder,'fs_RF.npy'), allow_pickle=True)
# fs_SVM=np.load(os.path.join(path,FS_folder,'fs_SVM.npy'), allow_pickle=True)
# selected_features_classificatore={'KNN':fs_KNN,'RF':fs_RF,'SVM':fs_SVM}


# selected_feature=[]
# for key in selected_features_classificatore.keys():
#     for f,feature in enumerate(feature_name_train):
#         if feature in selected_features_classificatore[key]:
#             if feature not in selected_feature:
#                 selected_feature.append(feature)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------decomment this line to analyse all the feature ------------------------------------------------------------------------------------------------
selected_feature=feature_name_train
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

condiction=np.zeros(len(feature_name_train))

for f,feature in enumerate(feature_name_train):

    if feature in selected_feature:
        condiction[f]=1

train_sel=train[:,condiction==1]

matrice_1=[] # for nuclei
matrice_0=[] # for vassels

for i, lb in enumerate(gt_train):
    if lb=='1.0':
        matrice_1.append(train_sel[i,:])
    else:
        matrice_0.append(train_sel[i,:])

matrice_1 = np.vstack(matrice_1)

matrice_0 = np.vstack(matrice_0)

# #used to analise the correlated feature 
gound_trut=[]
for i in gt_train:
    if i==1:
        gound_trut.append('tubuls')
    else:
        gound_trut.append('vassels')

#----------------------------------------------------- BOX PLOT ----------------------------------------------------------------------
# # Creazione del DataFrame con i dati 
# print(len(selected_feature))

# data = pd.DataFrame(train_sel, columns=selected_feature)
# data['Class'] = gound_trut

# # Melt del DataFrame per ottenere una colonna "Feature" e una colonna "Value"
# melted_data = pd.melt(data, id_vars='Class', var_name='Feature', value_name='Value')

# # Creazione del grafico dei box plot utilizzando Seaborn
# # plt.figure(figsize=(10, 6))
# plt.figure()
# sns.boxplot(data=melted_data, x='Feature', y='Value', hue='Class')

# plt.xticks(rotation=45)
# plt.xlabel('Features')
# plt.ylabel('Feature Values')


# plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------HEATMAP-------------------------------------------------------------
# data = pd.DataFrame(train_sel, columns=selected_feature)
# data['Class'] = gound_trut
# correlation_matrix = data.corr()
# # Creazione della heatmap
# plt.figure(figsize=(20, 20))
# sns.heatmap(correlation_matrix, annot=True, cmap='Reds', linewidths=0.5)
# plt.title('Heatmap of Feature Correlation')
# plt.show()
# #------------------------------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------STRIP PLOT-------------------------------------------------------------


# data = pd.DataFrame(train_sel, columns=selected_feature)
# data['Class'] = gound_trut
# # Melt del DataFrame per ottenere una colonna "Feature" e una colonna "Value"
# melted_data = pd.melt(data, id_vars='Class', var_name='Feature', value_name='Value')
# plt.figure()
# sns.stripplot(data=melted_data, x='Feature', y='Value', hue='Class')
# plt.xticks(rotation=45)
# plt.xlabel('Features')
# plt.ylabel('Feature Values')
# plt.show()

# #------------------------------------------------------------------------------------------------------------------------------------------

# #----------------------------------------------------------------------ISTOGRAM-------------------------------------------------------------
# for nf in range(train_sel.shape[1]):
#     data1=matrice_1[:,nf]
#     data0=matrice_0[:,nf]
#     name_feature=feature_name_train[nf]


#     #istogram

#     plt.hist(data1, bins=100, color='red', alpha=0.5, label='tubuls')
#     plt.hist(data0, bins=100, color='blue', alpha=0.5, label='vassels')
#     plt.xlabel('feature values')
#     plt.ylabel('frequency')
#     plt.title(f' {name_feature}' )
#     plt.legend()



#     plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------
