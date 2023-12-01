r''' code for the train, the validation and the test of the models
Autor Sonia'''
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage.morphology as sk
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm,metrics
import joblib
import tifffile as tiff
from bresenham import bresenham
from scipy import ndimage

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

def validation_SVM(c,degree,coef0,tol,kernel_names,train_sel,gt_train,validation_sel,gt_val):
    ''' function to implement the validation of the SVM classifier. I decide to let change four or three parameters in order to the chosen kernel. 
    For Poly kernel four and for the others three. the function test all the possible combination and save the valu of balance accuracy for each one. 
    the chosen model is the one that gives the higer balance accuracy on the validation set
    input:
    -c:list with the possible values for C
    -degree: degree of the polinomial function used just with poly kernel
    -coef0:list with the possible values for coef0
    -toll: list with the possible values for tol
    -kernel_name: list with the name of the kernel to test
    -train_sel: balanced and normalised train set cointaining only the chosen features
    -gt_train: the graund thrut for the train_sel
    -validation_sel: normalised validation set cointaining only the chosen features
    -get_val: GT for the validation_sel
    output:
    -b_accuracy_val: balanced accuracy of the winner combination
    -parameters: winner combination of parameters
    '''
    b_accuracy_val={}
    parameters={}
    for kernel in kernel_names:
        print(f'kernel: {kernel}')

        parameters[kernel]={}
        if kernel=='poly': # i have to divide the case in which i use the poly kernel because with this one i have to use four parameters
            # initialisation of the matrix in which i will put the values of balanced accuracy
            c_matrix=np.zeros(((len(degree)*len(coef0)*len(tol)),len(c))) 
            degree_matrix=np.zeros(((len(c)*len(coef0)*len(tol)),len(degree)))
            coef0_matrix=np.zeros(((len(degree)*len(c)*len(tol)),len(coef0)))
            tol_matrix=np.zeros(((len(degree)*len(coef0)*len(c)),len(tol)))
            cont_c=0
            cont_degree=0
            cont_coef0=0
            cont_tol=0

            # to upgraid the values of the coefficient
            jump_degree=len(coef0)*len(tol)
            k_degree=len(coef0)*len(tol)
            jump_coef0=len(tol)
            k_coef0=len(tol)

            for i,item_c in enumerate(c):
                print(f'C:{item_c}')

                for j,item_degree in enumerate(degree):
                    for k,item_coef0 in enumerate(coef0):
                        for z,item_tol in enumerate(tol):
                            classifier=svm.SVC(C=item_c,kernel=kernel,degree=item_degree,coef0=item_coef0,tol=item_tol) # training with the chosen paramentrs
                            classifier.fit(train_sel,gt_train)
                            y_predict_val=classifier.predict(validation_sel) # evaluation of the balance accuracy
                            b_accuracy=metrics.balanced_accuracy_score(gt_val,y_predict_val)
                            # all this line are to fill each matrix. the colon of the matrix rappresent all the values of the balance accuracy obtained when the paramenters has the values
                            # corresponding to the colon. (for istance if the higher balance accuracy is in the first colon of c_matrix it menas that the c values that generate that number is the first one in c list)
                            if cont_c>=c_matrix.shape[0]:
                                cont_c=0

                            c_matrix[cont_c,i]=b_accuracy
                            degree_matrix[cont_degree,j]=b_accuracy
                            coef0_matrix[cont_coef0,k]=b_accuracy
                            tol_matrix[cont_tol,z]=b_accuracy

                            if item_tol==tol[-1]:
                                cont_tol+=1
                            cont_c+=1          

                            if (cont_degree+1)%k_degree==0:
                                if item_degree==degree[-1]:
                                    cont_degree=jump_degree
                                    jump_degree+=k_degree
                                else:
                                    cont_degree=jump_degree-k_degree
                            else:
                                cont_degree+=1

                            if (cont_coef0+1)%k_coef0==0:
                                if item_coef0==coef0[-1]:
                                    cont_coef0=jump_coef0
                                    jump_coef0+=k_coef0
                                else:
                                    cont_coef0=jump_coef0-k_coef0
                            else:
                                cont_coef0+=1


            massimo=np.max(c_matrix) # find the max balance accuracy and the relative paramenters
            b_accuracy_val[kernel]=massimo
            parameters[kernel]['C']=c[np.where(c_matrix==massimo)[1][0]]
            parameters[kernel]['degree']=degree[np.where(degree_matrix==massimo)[1][0]]
            parameters[kernel]['coef0']=coef0[np.where(coef0_matrix==massimo)[1][0]]
            parameters[kernel]['tol']=tol[np.where(tol_matrix==massimo)[1][0]]

        else: # same as the first but just with three parametnrs 
            c_matrix=np.zeros(((len(coef0)*len(tol)),len(c)))
            coef0_matrix=np.zeros(((len(c)*len(tol)),len(coef0)))
            tol_matrix=np.zeros(((len(coef0)*len(c)),len(tol)))
            cont_c=0
            cont_coef0=0
            cont_tol=0

            jump_coef0=len(tol)
            k_coef0=len(tol)

            for i,item_c in enumerate(c):
                    print(f'C:{item_c}')
                    for k,item_coef0 in enumerate(coef0):
                        for z,item_tol in enumerate(tol):
                            classifier=svm.SVC(C=item_c,kernel=kernel,coef0=item_coef0,tol=item_tol)
                            classifier.fit(train_sel,gt_train)
                            y_predict_val=classifier.predict(validation_sel)
                            b_accuracy=metrics.balanced_accuracy_score(gt_val,y_predict_val)

                            if cont_c>=c_matrix.shape[0]:
                                cont_c=0

                            c_matrix[cont_c,i]=b_accuracy
                            coef0_matrix[cont_coef0,k]=b_accuracy
                            tol_matrix[cont_tol,z]=b_accuracy

                            if item_tol==tol[-1]:
                                cont_tol+=1
                            cont_c+=1          

                            if (cont_coef0+1)%k_coef0==0:
                                if item_coef0==coef0[-1]:
                                    cont_coef0=jump_coef0
                                    jump_coef0+=k_coef0
                                else:
                                    cont_coef0=jump_coef0-k_coef0
                            else:
                                cont_coef0+=1


            massimo=np.max(c_matrix)
            b_accuracy_val[kernel]=massimo
            parameters[kernel]['C']=c[np.where(c_matrix==massimo)[1][0]]
            parameters[kernel]['coef0']=coef0[np.where(coef0_matrix==massimo)[1][0]]
            parameters[kernel]['tol']=tol[np.where(tol_matrix==massimo)[1][0]]
        
        # print('done')

    return(b_accuracy_val,parameters)

def plot(class_tipe,output,prova,usable,patch_name,lume_labels_new):
    ''' function to plot the classification
    input:
    - class_tipe: string with the name of the classifier
    - output: output of the classifier
    - prova: copy of the RGB image
    -usable: list of usable lumen
    -patch_name
    -lume_labels_new: labels of the lumen in the lumen mask
    output:
    -prova: copy of the RGB image with the classified lumen colored in red if they are tubuls and in blu if they are vassels'''
    y_pred=output[class_tipe]
    for i,item in enumerate(usable):

            if item[1]==patch_name:

                if y_pred[i]=='1.0': # tubuls in red
            
                    prova[lume_labels_new==int(item[0])]=[255,0,0]
                    
                else:
                
                    prova[lume_labels_new==int(item[0])]=[0,0,255] # vassels in blu

    return(prova)
#%%



#%%
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

#---------------------------------------------------------decoment to implement the PCA FS----------------------------------------------------------------------------
    # print('----------PCA FS----------')
    # n_selected_feature=10
    # stardizer=StandardScaler()
    # matrice_pca=stardizer.fit_transform(train)
    # # matrice_pca=(matrice_tr-np.mean(matrice_tr))/np.std(matrice_tr)
    # pca = decomposition.PCA(n_components=n_selected_feature)
    # feat_PCA_rf = pca.fit_transform(matrice_pca)
    # pca_component = pca.components_
    # feat_PCA_pos=np.zeros(train.shape[1])
    # for component in pca_component:
    #     most_important_feature_index = np.abs(component).argmax()
    #     feat_PCA_pos[most_important_feature_index] = 1
    # cont=0
    # for ind,g in enumerate(feat_PCA_pos):
    #     if g==1:
    #         print(f'-{feature_name_train[ind]}')
    #         cont+=1
    # print(cont)

    # train_sel=train[:,feat_PCA_pos==1].astype(float)
    # validation_sel=validation[:,feat_PCA_pos==1].astype(float)
    # test_sel=test[:,feat_PCA_pos==1].astype(float)

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
    print(f'KNN\nValidation\n-chosen k: {k}\n-balance accuracy: {np.max(b_accuracy)}')
    chosen_classifier=classifier_dic[str(k)] # best model
    joblib.dump(chosen_classifier, os.path.join(path,models,'KNN_'+str(len(selected_feature))+'.pkl'))#saving the best model

#Testing
    winner_model['KNN']=chosen_classifier
    y_predict_test=chosen_classifier.predict(test_sel)
    output_test['KNN']= y_predict_test
    output_train['KNN']=chosen_classifier.predict(train_sel)


# Performance evaluation
    performance['KNN']={}
    accuracy_test=metrics.accuracy_score(gt_test,y_predict_test)
    b_accuracy_test=metrics.balanced_accuracy_score(gt_test,y_predict_test)
    tubul_precision_test=metrics.precision_score(gt_test,y_predict_test,pos_label='1.0')
    vassel_precision_test=metrics.precision_score(gt_test,y_predict_test,pos_label='0.0')
    tubul_recall_test=metrics.recall_score(gt_test,y_predict_test,pos_label='1.0')
    vassel_recall_test=metrics.recall_score(gt_test,y_predict_test,pos_label='0.0')


    performance['KNN']['accuracy']=accuracy_test
    performance['KNN']['balanced accuracy']=b_accuracy_test
    performance['KNN']['tubuls precision']=tubul_precision_test
    performance['KNN']['vassel precision']=vassel_precision_test
    performance['KNN']['tubuls recall']=tubul_recall_test
    performance['KNN']['vassel recall']=vassel_recall_test

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
    print(f'RF\nValidation\n-chosen number of trees: {t}\n-balance accuracy: {np.max(b_accuracy)}')
    chosen_classifier=classifier_dic[str(t)]
    joblib.dump(chosen_classifier, os.path.join(path,models,'RF_'+str(len(selected_feature))+'.pkl'))#saving the best model

# Testing
    winner_model['RF']=chosen_classifier
    y_predict_test=chosen_classifier.predict(test_sel)
    output_test['RF']= y_predict_test
    output_train['RF']=chosen_classifier.predict(train_sel)

# Performance evaluation
    performance['RF']={}
    accuracy_test=metrics.accuracy_score(gt_test,y_predict_test)
    b_accuracy_test=metrics.balanced_accuracy_score(gt_test,y_predict_test)
    tubul_precision_test=metrics.precision_score(gt_test,y_predict_test,pos_label='1.0')
    vassel_precision_test=metrics.precision_score(gt_test,y_predict_test,pos_label='0.0')
    tubul_recall_test=metrics.recall_score(gt_test,y_predict_test,pos_label='1.0')
    vassel_recall_test=metrics.recall_score(gt_test,y_predict_test,pos_label='0.0')

    performance['RF']['accuracy']=accuracy_test
    performance['RF']['balanced accuracy']=b_accuracy_test
    performance['RF']['tubuls precision']=tubul_precision_test
    performance['RF']['vassel precision']=vassel_precision_test
    performance['RF']['tubuls recall']=tubul_recall_test
    performance['RF']['vassel recall']=vassel_recall_test



#--------------------------------------- SVM ----------------------------------------------------------
# Validation
    # params for the validation
    kernel_names=['poly','rbf','sigmoid']
    c=np.array([0.01,0.1,1])
    degree=np.array([2,3,4,5])
    coef0=np.array([0.5,1,5,10,20,30])
    tol=np.array([1e-8,1e-6,1e-5,1e-4])
    validation
    accuracy_dic,params_dic=validation_SVM(c,degree,coef0,tol,kernel_names,train_sel,gt_train,validation_sel,gt_val)
    # chosing the best combination considering the balance accuracy on the validation
    max=0.001
    for keys in accuracy_dic.keys():
        if accuracy_dic[keys]>max:
            max=accuracy_dic[keys]
            kernel=keys
    
    params=params_dic[kernel]
    print(f'SVM\nValidation\n-chosen kernel: {kernel}\n-chosen params: {params}\n-balance accuracy: {max}')

#Training
    if kernel=='poly':
        classifier=svm.SVC(C=params['C'],kernel=kernel,degree=params['degree'],coef0=params['coef0'],tol=params['tol'])
    else:
        classifier=svm.SVC(C=params['C'],kernel=kernel,coef0=params['coef0'],tol=params['tol'])

    classifier.fit(train_sel,gt_train)
    winner_model['SVM']=classifier
    joblib.dump(classifier, os.path.join(path,models,'SVM_'+str(len(selected_feature))+'.pkl'))#saving the best model

#Testing
    y_predict_test=classifier.predict(test_sel)
    output_test['SVM']= y_predict_test
    output_train['SVM']=classifier.predict(train_sel)

# Performances evaluation
    performance['SVM']={}
    accuracy_test=metrics.accuracy_score(gt_test,y_predict_test)
    b_accuracy_test=metrics.balanced_accuracy_score(gt_test,y_predict_test)
    tubul_precision_test=metrics.precision_score(gt_test,y_predict_test,pos_label='1.0')
    vassel_precision_test=metrics.precision_score(gt_test,y_predict_test,pos_label='0.0')
    tubul_recall_test=metrics.recall_score(gt_test,y_predict_test,pos_label='1.0')
    vassel_recall_test=metrics.recall_score(gt_test,y_predict_test,pos_label='0.0')

    performance['SVM']['accuracy']=accuracy_test
    performance['SVM']['balanced accuracy']=b_accuracy_test
    performance['SVM']['tubuls precision']=tubul_precision_test
    performance['SVM']['vassel precision']=vassel_precision_test
    performance['SVM']['tubuls recall']=tubul_recall_test
    performance['SVM']['vassel recall']=vassel_recall_test

# ---------------------------------------------------------majority votuing (MV)-------------------------------------------------------------
    mv=[]
    for cl in range(len(output_test['KNN'])):
        t=0
        v=0

        for k in output_test.keys():
            if output_test[k][cl]=='1.0':
                t+=1
            else:
                v+=1
        if t>v:
            mv.append('1.0')
        else:
            mv.append('0.0')
        
    output_test['MV']=np.array(mv)

    y_predict_test= output_test['MV']

    performance['MV']={}
    accuracy_test=metrics.accuracy_score(gt_test,y_predict_test)
    b_accuracy_test=metrics.balanced_accuracy_score(gt_test,y_predict_test)
    tubul_precision_test=metrics.precision_score(gt_test,y_predict_test,pos_label='1.0')
    vassel_precision_test=metrics.precision_score(gt_test,y_predict_test,pos_label='0.0')
    tubul_recall_test=metrics.recall_score(gt_test,y_predict_test,pos_label='1.0')
    vassel_recall_test=metrics.recall_score(gt_test,y_predict_test,pos_label='0.0')

    performance['MV']['accuracy']=accuracy_test
    performance['MV']['balanced accuracy']=b_accuracy_test
    performance['MV']['tubuls precision']=tubul_precision_test
    performance['MV']['vassel precision']=vassel_precision_test
    performance['MV']['tubuls recall']=tubul_recall_test
    performance['MV']['vassel recall']=vassel_recall_test

#---------------------------------------------------------------PERFROMANCES-----------------------------------------------------------------------------
# evaluation of the performances on the single images to evaluate the standard deviation of the performances
    for_std={}
    performance_single_image={}

    for patch_name in sorted(os.listdir(os.path.join(path,big_folder,set_folder_test,patch_folder))):
        # standard deviation evauation
        patch_name=patch_name.split('.')[0]
        for_std[patch_name]={'gt':[],
                             'pred':{'KNN':[],'RF':[],'SVM':[],'MV':[]}}
        
    for patch_name in sorted(os.listdir(os.path.join(path,big_folder,set_folder_test,patch_folder))):
        patch_name=patch_name.split('.')[0]
        for  num,element in enumerate(usable_test):
            if element[1]==patch_name:
                for_std[patch_name]['gt'].append(gt_test[num])

        for key in output_test.keys():
            performance_single_image[key]={}
            performance_single_image[key]['accuracy']=[]
            performance_single_image[key]['balanced accuracy']=[]
            performance_single_image[key]['tubuls precision']=[]
            performance_single_image[key]['vassel precision']=[]
            performance_single_image[key]['tubuls recall']=[]
            performance_single_image[key]['vassel recall']=[]

            for  num,element in enumerate(usable_test):
                if element[1]==patch_name:
                        for_std[patch_name]['pred'][key].append( output_test[key][num])

    for nome_patch in for_std.keys():
        gt_single_image=for_std[nome_patch]['gt']
        for key in for_std[nome_patch]['pred'].keys():

            y_pred_single_image=for_std[nome_patch]['pred'][key]
            accuracy=metrics.accuracy_score(gt_single_image,y_pred_single_image)
            b_accuracy=metrics.balanced_accuracy_score(gt_single_image,y_pred_single_image)
            tubul_precision=metrics.precision_score(gt_single_image,y_pred_single_image,pos_label='1.0')
            vassel_precision=metrics.precision_score(gt_single_image,y_pred_single_image,pos_label='0.0')
            tubul_recall=metrics.recall_score(gt_single_image,y_pred_single_image,pos_label='1.0')
            vassel_recall=metrics.recall_score(gt_single_image,y_pred_single_image,pos_label='0.0')

            performance_single_image[key]['accuracy'].append(accuracy)
            performance_single_image[key]['balanced accuracy'].append(b_accuracy)
            performance_single_image[key]['tubuls precision'].append(tubul_precision)
            performance_single_image[key]['vassel precision'].append(vassel_precision)
            performance_single_image[key]['tubuls recall'].append(tubul_recall)
            performance_single_image[key]['vassel recall'].append(vassel_recall)


    #printing of the performances
    print('TEST PERFORMANCES')
    for classificator_type in performance_single_image.keys():
        print(f'{classificator_type}')
        for key in performance[classificator_type].keys():
            print(f'-{key}: {performance[classificator_type][key]*100} %  +/- {np.std(performance_single_image[classificator_type][key])*100}%')


#--------------------------------------------------------------------RESULT SHOWING----------------------------------------------------------------------------

    for patch_name in sorted(os.listdir(os.path.join(path,big_folder,set_folder_test,patch_folder))):

        patch_name=patch_name.split('.')[0]
        image_rgb = np.array(Image.open(
                os.path.join(path,big_folder,set_folder_test,patch_folder,patch_name+'.jpg')
        ))

        image=np.array(Image.open(
                os.path.join(path,big_folder,set_folder_test,patch_folder,patch_name+'.jpg')
        ).convert('L'))


        # LUMEN DETECTION with threshold
        lumen_mask_th=lumen_detecion_thresholding(np.asarray(image),lume_size_up_th,lume_size_down_th,area_holes)
        
        # channel division e combination
        r_g=abs(np.asarray(image_rgb)[:,:,0].astype(int)-np.asarray(image_rgb)[:,:,1].astype(int)).astype(np.uint8)
        r_g_c=contrast_augmentation(r_g,1).astype(np.uint8)

        # improving lumen mask with the aggregation of the smaller mask that belong to the same lumen
        lume_labels_new=mask_aggregation(r_g_c,lumen_mask_th)

        gt_image=image_rgb.copy()
        propro=np.zeros_like(image)
        for num,element in enumerate(usable_test):
            propro[lume_labels_new==int(element[0])]=num
            if element[1]==patch_name:
                if gt_test[num]=='1.0':
                    gt_image[lume_labels_new==int(element[0])]=[255,0,0]
                else: 
                    gt_image[lume_labels_new==int(element[0])]=[0,0,255]

        nuclei_mask=tiff.imread(os.path.join(path,nuclei_mask_folder,patch_name+'_NucXav.tif'))
        pro=image_rgb.copy()
        pro[nuclei_mask>0]=[0,0,0]
        keys=output_test.keys()

        plt.figure()
        for k,key in enumerate(keys):
            if k+2>=4:
                a=k+3
            else:
                a=k+2
            out_image=image_rgb.copy()
            out_image=plot(key,output_test,out_image,usable_test,patch_name,lume_labels_new)
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
