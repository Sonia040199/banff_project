r''' 
to implement the classification on new images without the ground thruth
Author Sonia 
'''
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage.morphology as sk
import cv2
import joblib
from bresenham import bresenham
from scipy import ndimage


def contrast_augmentation(im, perc):
    ''' function to increse the contrast
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
    '''     function to obtain the lumen mask
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
    '''    function to connect the mask from the same lumen
    input:
    - r_g_c= channel of the red- channel of the green after the constrast aumentation 
    - lumen_mask_th: lumen mask
    output:
    - lume_labels_new = new image in which is associated a label of each lumen
 '''

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

if __name__=='__main__':

    # folders names
    path = os.getcwd()
    big_folder='DATA_SET'
    set_folder_new='new_images' # INSERT HERE THE NAME OF THE FOLDER IN WHICH THERE ARE THE NEW IMAGES
    patch_folder='patches'
    nuclei_mask_folder='nuclei_mask'
    models='trained_model'


    #constant definition
    lume_size_down_th= 965.2666367999998/2
    lume_size_up_th=229638.144
    area_holes=120


    set_folder=set_folder_new 

    matrice=np.load(os.path.join(path,big_folder,set_folder,'matrice.npy'), allow_pickle=True)
    feature_name=np.load(os.path.join(path,big_folder,set_folder,'feature_name.npy'), allow_pickle=True)
    usable=np.load(os.path.join(path,big_folder,set_folder,'usable.npy'), allow_pickle=True)

    selected_feature=['lumen_size','width_lumen','hight_lumen','lumen_eccentricity','convex_hull','number_nuclei','mean_nuclei_size','std_nuclei_size','mean_nuclei_shape' ,'std_nuclei_shape'] 

    #KNN
    condiction=np.zeros(len(feature_name))
    for f,feature in enumerate(feature_name):
        if feature in selected_feature:
            print(f'-{feature}')
            condiction[f]=1

    test_sel=matrice[:,condiction==1].astype(float)

    output={}
 

    #KNN
    classifier_knn = joblib.load(os.path.join(path,models,'KNN_'+str(len(selected_feature))+'.pkl')) #load of the trained model
    y_predict=classifier_knn.predict(test_sel)
    output['KNN']= y_predict

    #RF
    classifier_rf= joblib.load(os.path.join(path,models,'RF_'+str(len(selected_feature))+'.pkl')) #load of the trained model
    y_predict=classifier_rf.predict(test_sel)
    output['RF']= y_predict





    #SVM
    classifier_svm= joblib.load(os.path.join(path,models,'SVM_'+str(len(selected_feature))+'.pkl')) #load of the trained model
    y_predict=classifier_svm.predict(test_sel)
    output['SVM']= y_predict


    #-------------------------------------------------------------- majority votuing--------------------------------------------------------------
    mv=[]
    for cl in range(len(output['KNN'])):
        t=0
        v=0

        for k in output.keys():
            if output[k][cl]=='1.0':
                t+=1
            else:
                v+=1
        if t>v:
            mv.append('1.0')
        else:
            mv.append('0.0')
        
    output['MV']=np.array(mv)

    y_predict= output['MV']


    # plot 

    for patch_name in sorted(os.listdir(os.path.join(path,big_folder,set_folder,patch_folder))):
            patch_name=patch_name.split('.')[0]

            # RGB image
            image_rgb = np.array(Image.open(
                        os.path.join(path,big_folder,set_folder,patch_folder,patch_name+'.jpg')
                ))
            #gray scale image
            image=np.array(Image.open(
                        os.path.join(path,big_folder,set_folder,patch_folder,patch_name+'.jpg')
                ).convert('L'))


            # LUMEN DETECTION with threshold
      
            lumen_mask_th=lumen_detecion_thresholding(np.asarray(image),lume_size_up_th,lume_size_down_th,area_holes)
            
            # channel division e combination
            r_g=abs(np.asarray(image_rgb)[:,:,0].astype(int)-np.asarray(image_rgb)[:,:,1].astype(int)).astype(np.uint8)
            r_g_c=contrast_augmentation(r_g,1).astype(np.uint8)

            # improving lumen mask with the aggregation of the smaller mask that belong to the same lumen
            lume_labels_new=mask_aggregation(r_g_c,lumen_mask_th)

            keys=output.keys()

            plt.figure()
            for k,key in enumerate(keys):
                out_image=image_rgb.copy()
                out_image=plot(key,output,out_image,usable,patch_name,lume_labels_new)
                plt.subplot(2,2,k+1)
                plt.imshow(out_image)
                plt.title(key)
                plt.axis('off')
        
            plt.show()