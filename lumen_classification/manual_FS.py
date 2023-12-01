r''' code to implement the manual FS comparing the balance accuracy of the model applied to the train set. the code save the perfromances for each classifier and cosiderinf
one feature per time in an excel file. Is also able to select a set of feature for each classifier excluding the features that give performances lower than the previous one.
the selected features are saved in a specific folder. 
Author Sonia '''
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage.morphology as sk
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm,metrics
from bresenham import bresenham
from scipy import ndimage
import pandas as pd


def contrast_augmentation(im, perc):
    '''function to increse the contrast
    input:
    - im= image tomodify
    - perc= percentage of augmentatio. ( if perc=1 there is an augmentation of the 100%)
    output:
    - im_adjusted = modified image'''

    m = np.median(im)*0.95 # to decrese the number of pixel that are undred the threshold i had chosen to use the 95% of the luminance. 
    delta = perc * m
    im_adjusted = np.clip(im - delta, 0, 255).astype(np.uint8)
    return im_adjusted


def lumen_detecion_thresholding(image,lume_size_up_th,lume_size_down_th,area):
    '''function to obtain the lumen mask
    input:
    - image= image gray scale 
    - lume_size_up_th: max dimensione that a mask can have to be considered
    - lume_size_down_th: min dimensione that a mask can have to be considered
    - area: dimensione of the hole to remove
    output:
    - lumen_mask_fil = lumen mask after the post processing'''

    # PRE PROCESSING
    image_c=contrast_augmentation(image,1) # contrast aumentation

    # GLOBAL THRESHOLDING
    # inizialisation 
    lumen_mask = np.zeros_like(image_c, dtype=np.uint8)
    # threshold identification
    th, _= cv2.threshold(image_c, np.min(image_c), np.max(image_c), cv2.THRESH_BINARY + cv2.THRESH_OTSU) # appling the Otzu threshold to image to identify the TH for the segmentation
    lumen_mask=np.where(image_c>=th,1,lumen_mask) # creation of the mask


    # POST PROCESSING
    ## find contourns and fill the intern
    contours = cv2.findContours(np.uint8(lumen_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    a=contours[0]
    lumen_mask_fil=np.zeros_like(lumen_mask)
    cv2.drawContours(lumen_mask_fil,a,-1,1,cv2.FILLED)

    ## find the connected component in the binary image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lumen_mask_fil)
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA])+1
    if largest_label==1 or stats[int(largest_label), cv2.CC_STAT_AREA]>lume_size_up_th : 
        lumen_mask_fil[labels == largest_label] = 0  # check about the size of the bigger mask

    for i in range(num_labels):
        if stats[i, cv2.CC_STAT_AREA]<lume_size_down_th:     # elimination of the mask with a size smaller than lume_size_down_th to clean the mask
            lumen_mask_fil[labels == i] = 0
        if stats[i, cv2.CC_STAT_AREA]>lume_size_up_th: # elimination of the mask with a size bigger than lume_size_up_th to remove the back ground
            lumen_mask_fil[labels == i] = 0

    # fill the small holes in the images
    lumen_mask_fil=sk.remove_small_holes(lumen_mask_fil,area)
    return(lumen_mask_fil)

def mask_aggregation(r_g_c,lumen_mask_th):
    '''function to connect the mask from the same lumen
    input:
    - r_g_c= channel of the red- channel of the green after the constrast aumentation 
    - lumen_mask_th: lumen mask
    output:
    - lume_labels_new = new image in which is associated a label of each lumen '''

    lume_num_labels, lume_labels, _, _ = cv2.connectedComponentsWithStats(np.uint8(lumen_mask_th)) # find the connected component in the binary image
    alpha=(np.mean(r_g_c[lumen_mask_th==1])*2)# value with which I must confrotnate the pixels between the centroids
    centroidi=[] # list in which I put all the centroids for each mask

# calculation of centroids for each mask and filling the centroid list
    for i in range(lume_num_labels):
        if i==0: # the firt label is associated to all the mask
            continue
        lumen_i=np.zeros_like(lumen_mask_th) 
        lumen_i[lume_labels==i]=1 # lumen_i rappresnet the i-th lumen
        centroid = ndimage.measurements.center_of_mass(lumen_i)
        centroidi.append((int(centroid[1]),int(centroid[0])))

    new_lumen_labels=np.zeros(len(centroidi)) # list to keep track of lumen labels

    label_i=1 
    couple_done=[]

    for pos1,centroide_1 in enumerate(centroidi): # for each centroid in the list
        if new_lumen_labels[pos1]==0: # if the centroid has not yet been assigned a value
            new_lumen_labels[pos1]=label_i # I assign the current value of label_i and increment it
            label_i+=1
        for pos2,centroide_2 in enumerate(centroidi): # cycle to compare all other centroids in the region with the i-th
            if centroide_1==centroide_2 :
                continue
            if (centroide_1,centroide_2) in couple_done:
                continue

            couple_done.append((centroide_1,centroide_2))
            couple_done.append((centroide_2,centroide_1))

            values=[] 
            point_line=list(bresenham(centroide_1[0], centroide_1[1],centroide_2[0], centroide_2[1])) # evaluation of the bresenham distance between the centroid
            for point in point_line:
                values.append(r_g_c[point[1],point[0]])
            if np.all(np.array(values)<=alpha): # if there are no membranes between the centroids the one I am comparing takes the label of the former
                a=new_lumen_labels[pos2]
                new_lumen_labels[pos2]=new_lumen_labels[pos1] # the two centroids belong to the same lumen so they must have the same label
                for k in range(len(new_lumen_labels)): # decreasing the values of all other values as I deleted an element
                    if new_lumen_labels[k]>a:
                        new_lumen_labels[k]=new_lumen_labels[k]-1
            else: #Otherwise, if there are memebranes and the centroid does not yet have a value, I will assign and increment it
                if new_lumen_labels[pos2]==0:
                    new_lumen_labels[pos2]=label_i
                    label_i+=1

    lume_labels_new=np.zeros_like(lume_labels)
    for i in range(lume_num_labels):
        if i==0:
            continue
        lume_labels_new[lume_labels==i]=new_lumen_labels[i-1]

    return(lume_labels_new)


if __name__=='__main__':

    # folders names
    path = os.getcwd()
    big_folder='DATA_SET'
    set_folder_val='VALIDATION'
    set_folder_train='TRAIN'
    set_folder_test='TEST'
    patch_folder='patches'
    nuclei_mask_folder='nuclei_mask'
    excel_file='lumen_labels'
    models='trained_model'
    FS_folder='feature_selected'
 

    #constant definition
    lume_size_down_th= 965.2666367999998/2
    lume_size_up_th=229638.144
    area_holes=120
    output_test={}    
    output_train={}

    # train load
    train=np.load(os.path.join(path,big_folder,set_folder_train,'train.npy'), allow_pickle=True)
    gt_train=np.load(os.path.join(path,big_folder,set_folder_train,'gt_train.npy'), allow_pickle=True)
    usable_train=np.load(os.path.join(path,big_folder,set_folder_train,'usable_train.npy'), allow_pickle=True)
    feature_name_train=np.load(os.path.join(path,big_folder,set_folder_train,'feature_name.npy'), allow_pickle=True)


    # validation load
    validation=np.load(os.path.join(path,big_folder,set_folder_val,'validation.npy'), allow_pickle=True)
    gt_val=np.load(os.path.join(path,big_folder,set_folder_val,'gt_val.npy'), allow_pickle=True)
    usable_val=np.load(os.path.join(path,big_folder,set_folder_val,'usable_val.npy'), allow_pickle=True)


    # test load
    test=np.load(os.path.join(path,big_folder,set_folder_test,'test.npy'), allow_pickle=True)
    gt_test=np.load(os.path.join(path,big_folder,set_folder_test,'gt_test.npy'), allow_pickle=True)
    usable_test=np.load(os.path.join(path,big_folder,set_folder_test,'usable_test.npy'), allow_pickle=True)
 

    np.random.seed(20) # fix some seeds to garantee always the same performances
    b_KNN=[]
    b_RF=[]
    b_SVM=[]
    # to save the selected feature
    fs_KNN=[]
    fs_SVM=[]
    fs_RF=[]
    for f,s_feature in enumerate(feature_name_train):
        selected_feature=feature_name_train[f]
        print(f'\nFEATURE {f+1}/{len(feature_name_train)}: {selected_feature}')
     
        condiction=np.zeros(len(feature_name_train))
        condiction[f]=1

        train_sel=train[:,condiction==1].astype(float)
        validation_sel=validation[:,condiction==1].astype(float)
        test_sel=test[:,condiction==1].astype(float)
        winner_model={}
#--------------------------------------- KNN----------------------------------------------------------
    # Validation
        k_values=[3,5,7,15,25,35,45,55,65,75,85,95,105] # values of k for the validation
        b_accuracy=[]
        classifier_dic={}

        # train of the models and application on the validation
        for k in k_values:
            classifier= KNeighborsClassifier(n_neighbors=k)
            # training
            classifier.fit(train_sel,gt_train)
            classifier_dic[str(k)]=classifier
            y_predict_val=classifier.predict(validation_sel)
            b_accuracy.append(metrics.balanced_accuracy_score(gt_val,y_predict_val))
        
        # chosing the best model considering the balance accuracy on the validation
        k=k_values[np.where(b_accuracy==np.max(b_accuracy))[0][0]]
        chosen_classifier=classifier_dic[str(k)] # best model

    #Testing
        winner_model['KNN']=chosen_classifier
        y_predict_test=chosen_classifier.predict(test_sel)
        output_test['KNN']= y_predict_test

        # application on the train set
        y_predict_train=chosen_classifier.predict(train_sel)
        b_accuracy_train=metrics.balanced_accuracy_score(gt_train,y_predict_train)
        b_KNN.append(b_accuracy_train)
        print(f'knn:{b_accuracy_train}')

        #iterative selection of the feature
        if len(b_KNN)==1: # if is the first feature
            fs_KNN.append(s_feature)
        else:
            # i consider just the feature that give me and increase in the performances on the train set
            if b_accuracy_train>=b_KNN[-2]: 
                fs_KNN.append(s_feature)

    #--------------------------------------- RF----------------------------------------------------------
    # Validation
        t_values=[20,70,50,100,150,200] # number of tree for the validation
        b_accuracy=[]
        classifier_dic={}
        # train of the models and application on the validation
        for t in t_values:
            classifier= RandomForestClassifier(n_estimators=t)
    # Training
            classifier.fit(train_sel,gt_train)
            classifier_dic[str(t)]=classifier
            y_predict_val=classifier.predict(validation_sel)
            b_accuracy.append(metrics.balanced_accuracy_score(gt_val,y_predict_val))
        # chosing the best model considering the balance accuracy on the validation
        t=t_values[np.where(b_accuracy==np.max(b_accuracy))[0][0]]
        chosen_classifier=classifier_dic[str(t)]
    # Testing
        winner_model['RF']=chosen_classifier
        y_predict_test=chosen_classifier.predict(test_sel)
        output_test['RF']= y_predict_test

        # application on the train set
        y_predict_train=chosen_classifier.predict(train_sel)
        b_accuracy_train=metrics.balanced_accuracy_score(gt_train,y_predict_train)
        b_RF.append(b_accuracy_train)
        print(f'rf:{b_accuracy_train}')

        if len(b_RF)==1:
            fs_RF.append(s_feature)
        else:
            if b_accuracy_train>=b_RF[-2]:
                fs_RF.append(s_feature)

    #--------------------------------------- SVM ----------------------------------------------------------
    # considering the computation time i fix the paramenters for the SVM the chosen feature are always the same
        params={'C': 0.1, 'degree': 4, 'coef0': 20.0, 'tol': 1e-08}
        kernel='poly'
    #Training
        if kernel=='poly':
            classifier=svm.SVC(C=params['C'],kernel=kernel,degree=params['degree'],coef0=params['coef0'],tol=params['tol'])
        else:
            classifier=svm.SVC(C=params['C'],kernel=kernel,coef0=params['coef0'],tol=params['tol'])
        classifier.fit(train_sel,gt_train)
        winner_model['SVM']=classifier

    #Testing
        y_predict_test=classifier.predict(test_sel)
        output_test['SVM']= y_predict_test

        # application on the train set
        y_predict_train=classifier.predict(train_sel)
        b_accuracy_train=metrics.balanced_accuracy_score(gt_train,y_predict_train)
        b_SVM.append(b_accuracy_train)
        print(f'svm:{b_accuracy_train}')

        if len(b_SVM)==1:
            fs_SVM.append(s_feature)
        else:
            if b_accuracy_train>=b_SVM[-2]:
                fs_SVM.append(s_feature)


    # saving of the selected feature for each classifier
    np.save(os.path.join(path,FS_folder,'fs_KNN.npy'), fs_KNN)
    np.save(os.path.join(path,FS_folder,'fs_RF.npy'), fs_RF)
    np.save(os.path.join(path,FS_folder,'fs_SVM.npy'), fs_SVM)

# save the data in a excel sheet to visualise them
    feature_selection_excel='feature_selection'

    dati = {
        "feature_name": feature_name_train,
        "KNN": b_KNN,
        "RF": b_RF,
        "SVM": b_SVM
    }

    # save the data in a excel sheet to visualise them
    df = pd.DataFrame(dati)
    writer = pd.ExcelWriter(os.path.join(path,feature_selection_excel+'.xlsx'), engine='openpyxl')
    df.to_excel(writer, sheet_name="FS_selection", index=False)
    writer.save()

    
