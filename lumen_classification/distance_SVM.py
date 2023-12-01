r''' the idea is to have the distance between each element and the plan used by SVM model. In this way is possible to define a threshold to apply to the distance to reduce
the number of non correct classified 
Author Sonia '''


import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage.morphology as sk
import cv2
from sklearn import svm,metrics


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



#--------------------------------------------------------FEATURE SELECTION---------------------------------------------------------

#------------------------------------------------------decoment to implement the manual FS--------------------------------------------------------------------
    selected_feature=['lumen_size','width_lumen','hight_lumen','lumen_eccentricity','convex_hull','number_nuclei','mean_nuclei_size','std_nuclei_size','mean_nuclei_shape' ,'std_nuclei_shape']
    
    condiction=np.zeros(len(feature_name_train))
    for f,feature in enumerate(feature_name_train):
        if feature in selected_feature:
            print(f'-{feature}')
            condiction[f]=1

    train_sel=train[:,condiction==1].astype(float) # train set after feature selection
    validation_sel=validation[:,condiction==1].astype(float) # validation set after feature selection
    test_sel=test[:,condiction==1].astype(float) # test set after feature selection

    winner_model={}
    performance={}



    # SVM
    
    params={'C': 0.1, 'degree': 3, 'coef0': 1, 'tol': 1e-08} # define the params values chosen during the validation
    kernel='poly'


    if kernel=='poly':
        classifier=svm.SVC(C=params['C'],kernel=kernel,degree=params['degree'],coef0=params['coef0'],tol=params['tol'])
    else:
        classifier=svm.SVC(C=params['C'],kernel=kernel,coef0=params['coef0'],tol=params['tol'])

    classifier.fit(train_sel,gt_train)
    y_pred_val=classifier.predict(validation_sel)
    b_accuracy_val=metrics.balanced_accuracy_score(gt_val,y_pred_val)
    print(f'SVM\nValidation\n-chosen kernel: {kernel}\n-chosen params: {params}\n-balance accuracy: {b_accuracy_val}')

    # Apply the model on the Test set
    y_predict_test=classifier.predict(test_sel)
    distance=classifier.decision_function(test_sel) # extract the distance between each element and the iper-plan 

    average_distance_erros=[] # to collect the average distance of the non correct classified
    average_distance_good=[] # to collect the average distance of the correct classified
    errors=0 # to count the number of errors

    for i,item in enumerate(y_predict_test):
        if item!=gt_test[i]:
            errors+=1
            average_distance_erros.append(distance[i])
            print(f'-real class: {gt_test[i]}\n-predicted class: {item}\n-distance: {distance[i]}')
        else:
            average_distance_good.append(distance[i])
    
    print(f'total number of erros: {errors}\n-average NON CORRECT classified distance: {np.mean(average_distance_erros)}')
    print(f'-average distance CORRECT classified: {np.mean(average_distance_good)}')