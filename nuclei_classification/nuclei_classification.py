r'''
For the classification of the nuclei. It saves the trained model in the folder models
Author Sonia 
'''
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import tifffile as tiff

from PIL import Image
from sklearn import svm,metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def cambio_gt(gt):
    ''' fucntion to trasform the gt in string following the output of the models'''

    gt_new=[]
    for i in gt:
        if i==1:
            gt_new.append('1.0')
        else:
            gt_new.append('0.0')
    return(np.array(gt_new))

def classificatore(train,test,gt_train,gt_test,classifier,class_type,path,folder_model,output_test,output_train):
    ''' function to train all the model
    input:
    - train: train set
    - test: test set
    -gt_train: GT of the train set
    -gt_test: GT of the test set
    -classifier: not trained model
    -class_type: string with the name of the kind of classifier
    -path: path in which there is the code
    - folder_model: name of the folder where the trained model has to be saved
    - output_test and ptput_train: dictionary where the prediction has to be saved
    output
    -upgraited output_train and output_test'''
    # training
    classifier.fit(train,gt_train) 
    joblib.dump(classifier, os.path.join(path,folder_model,class_type+'.pkl')) # saving the model
    # testing
    y_predict_test=classifier.predict(test)
    y_predict_train=classifier.predict(train)
    output_train[class_type]=y_predict_train
    output_test[class_type]=y_predict_test
    #performances evaluation
    accuracy=metrics.accuracy_score(gt_test,y_predict_test)
    b_accuracy=metrics.balanced_accuracy_score(gt_test,y_predict_test)
    print(f'- accuracy= {accuracy}        balanced accuracy= {b_accuracy}\n')
    return(output_train,output_test)


def plot(class_tipe,output,prova,with_crown,patch_name,nuclei_mask):
    '''function to plot the prediction'''
    y_pred=output[class_tipe]
    for i in enumerate(with_crown):
            if i[1][1]==patch_name:
                nuclei_i=np.zeros_like(nuclei_mask)
                nuclei_i[nuclei_mask==int(i[1][0])]=1

                if y_pred[i[0]]=='1.0': # tubuls in red
            
                    prova[nuclei_i==1]=[255,0,0]
                    
                else:
                
                    prova[nuclei_i==1]=[0,0,255] # vassels in blu

    return(prova)
if __name__=='__main__':
    # folders names
    path = os.getcwd()
    big_folder='DATA_SET'
    set_folder_train='TRAIN'
    set_folder_test='TEST'
    patch_folder='patches'
    nuclei_mask_folder='nuclei_mask'
    excel_file='nuclei_lables'
    folder_model='models'

    output_test={}
    output_train={}

    # train load
    train=np.load(os.path.join(path,big_folder,set_folder_train,'train.npy'), allow_pickle=True)
    gt_train_int=np.load(os.path.join(path,big_folder,set_folder_train,'gt_train.npy'), allow_pickle=True)
    gt_train=cambio_gt(gt_train_int)
    with_crown_train=np.load(os.path.join(path,big_folder,set_folder_train,'usable_train.npy'), allow_pickle=True)
    feature_name_train=np.load(os.path.join(path,big_folder,set_folder_train,'feature_name.npy'), allow_pickle=True)

    # test load
    test=np.load(os.path.join(path,big_folder,set_folder_test,'test.npy'), allow_pickle=True)
    gt_test=np.load(os.path.join(path,big_folder,set_folder_test,'gt_test.npy'), allow_pickle=True)
    with_crown_test=np.load(os.path.join(path,big_folder,set_folder_test,'usable_test.npy'), allow_pickle=True)

    # #FEATURE SELECTION
    # condiction=np.zeros(len(feature_name_train))
    # for f,feature in enumerate(feature_name_train):
    #     if feature in selected_feature:
    #         print(f'-{feature}')
    #         condiction[f]=1

    # train_sel=train[:,condiction==1].astype(float) # train set after feature selection
    # test_sel=test[:,condiction==1].astype(float) # test set after feature selection



    print('\n----------------SVM---------------')
    classifier_svm= svm.SVC()
    output_train,output_test=classificatore(train,test,gt_train,gt_test,classifier_svm,'SVM',path,folder_model,output_test,output_train)

    print('\n---------------KNN----------------')
    classifier_KNN= KNeighborsClassifier()
    output_train,output_test=classificatore(train,test,gt_train,gt_test,classifier_KNN,'KNN',path,folder_model,output_test,output_train)

    print('\n---------------MLP----------------')
    classifier_MLP= MLPClassifier()
    output_train,output_test=classificatore(train,test,gt_train,gt_test,classifier_MLP,'MLP',path,folder_model,output_test,output_train)

    print('\n---------------RF----------------')
    classifier_RF= RandomForestClassifier()
    output_train,output_test=classificatore(train,test,gt_train,gt_test,classifier_RF,'RF',path,folder_model,output_test,output_train)



    for patch_name in sorted(os.listdir(os.path.join(path,big_folder,set_folder_test,patch_folder))):
        
        patch_name=patch_name.split('.')[0]
        image_rgb = np.array(Image.open(
                os.path.join(path,big_folder,set_folder_test,patch_folder,patch_name+'.jpg')
        ))

        image=np.array(Image.open(
                os.path.join(path,big_folder,set_folder_test,patch_folder,patch_name+'.jpg')
        ).convert('L'))

        nuclei_mask=tiff.imread(os.path.join(path,nuclei_mask_folder,patch_name+'_NucXav.tif'))


        
        gt_image=image_rgb.copy()
        for num,element in enumerate(with_crown_test):
            if element[1]==patch_name:
                if gt_test[num]=='1.0':
                    gt_image[nuclei_mask==int(element[0])]=[255,0,0]
                else: 
                    gt_image[nuclei_mask==int(element[0])]=[0,0,255]

        key=output_test.keys()
        plt.figure()
        for k,key in enumerate(key):
            if k+2>=4:
                a=k+3
            else:
                a=k+2
            out_image=image_rgb.copy()
            out_image=plot(key,output_test,out_image,with_crown_test,patch_name,nuclei_mask)
            plt.subplot(2,3,1)
            plt.imshow(gt_image)
            plt.title('GT')
            plt.axis('off')
            plt.subplot(2,3,a)
            plt.imshow(out_image)
            plt.title(key)
            plt.axis('off')
            plt.subplot(2,3,4)
            plt.imshow(image_rgb)
            plt.title('original')
            plt.axis('off')
       
        plt.show()

# %%
