r''' CODE TO CHECK THE GIVEN LABELS IN THE EXCEL FILE
for each image is rappresented a plot with the actual GT. the lables from the patologist and the labels of the mask. 
So checking the labels in the mask labels image is possible to modify the raw corresponding to the patch in the excel file.
Author Sonia '''

import openpyxl
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tifffile as tiff
import cv2
import skimage.morphology as sk
from scipy import ndimage
from bresenham import bresenham



def contrast_augmentation(im, perc):
    '''function to increse the contrast
    input:
    - im= image tomodify
    - perc= percentage of augmentatio. ( if perc=1 there is an augmentation of the 100%)
    output:
    - im_adjusted = modified image '''
    
    m = np.median(im)*0.95 # to decrese the number of pixel that are undred the threshold i had chosen to use the 95% of the luminance. 
    delta = perc * m
    im_adjusted = np.clip(im - delta, 0, 255).astype(np.uint8)
    return im_adjusted


def lumen_detecion_thresholding(image,lume_size_up_th,lume_size_down_th,area):
    ''' function to obtain the lumen mask
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


# folder name definition
path = os.getcwd()
big_folder='DATA_SET'
set_folder_val='VALIDATION'
set_folder_train='TRAIN'
set_folder_test='TEST'
set_folder_tri_staining='TRI_staining'
set_folder_fake_pas='FAKE_PAS_staining'
set_folder_bad_image='difficult'
set_folder_new='new_images'
patch_folder='patches'
nuclei_mask_folder='nuclei_mask'
excel_file='lumen_labels'
label_folder='label'


#constant definition
lume_size_down_th= 965.2666367999998/2
lume_size_up_th=229638.144
area_holes=120

image_done=[]

# definition of the data set to analyse
set_folder=set_folder_val

excel=openpyxl.load_workbook(os.path.join(path,excel_file+'.xlsx')) # open the excel file with the lables
sheet=excel.get_sheet_by_name('Sheet1')
# extraction of the row of the file
for i in range(sheet.max_row):
    if not isinstance(sheet[i+1][0].value,str):
        continue
    row_values = []
    for cell in sheet[i+1]:
        row_values.append(cell.value)
    patch_name=row_values[0]

    if patch_name in image_done:
        continue


    if patch_name+'.jpg' not in os.listdir(os.path.join(path,big_folder,set_folder,patch_folder)):
        continue
    print(patch_name)
    


    #loading of the GT image
    labeled_image = np.array(Image.open(
                    os.path.join(path,label_folder,patch_name+'.jpg')
            ))

    image_rgb = np.array(Image.open(
            os.path.join(path,big_folder,set_folder,patch_folder,patch_name+'.jpg')
    ))
    pro=image_rgb.copy()
    #gray scale image
    image=np.array(Image.open(
            os.path.join(path,big_folder,set_folder,patch_folder,patch_name+'.jpg')
    ).convert('L'))

    #-----------------------------------------------------------just when set_folder=set_folder_fake_pas--------------------------------------------------------------------------------
    # # RGB image
    # image_rgb = np.array(Image.open(
    #             os.path.join(path,main_folder,big_folder,set_folder,patch_folder,patch_name+'.png')
    #     ))
    # pro=image_rgb.copy()
    
    # #gray scale image
    # image=np.array(Image.open(
    #             os.path.join(path,main_folder,big_folder,set_folder,patch_folder,patch_name+'.png')
    #     ).convert('L'))

    # # loading of the GT image
    # patch_name_lab=patch_name.split('_')[0]+'_'+patch_name.split('_')[1]+'_'+patch_name.split('_')[2]
    # labeled_image = np.array(Image.open(
    #             os.path.join(path,main_folder,label_folder,patch_name_lab+'.jpg')
    #     ))
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# LUMEN DETECTION with threshold
    lumen_mask_th=lumen_detecion_thresholding(np.asarray(image),lume_size_up_th,lume_size_down_th,area_holes)
    lume_num_labels, lume_labels, _, _ = cv2.connectedComponentsWithStats(np.uint8(lumen_mask_th))

    r_g=abs(np.asarray(image_rgb)[:,:,0].astype(int)-np.asarray(image_rgb)[:,:,1].astype(int)).astype(np.uint8)
    r_g_c=contrast_augmentation(r_g,1).astype(np.uint8)

    lume_labels_new=mask_aggregation(r_g_c,lumen_mask_th)
    lume_num_label_new=np.max(lume_labels_new)
    

    for i in range(lume_num_label_new+1): # rappresentation of the given labels
        if i==0:
            continue
        if row_values[i]==0 or row_values[i]==0.5: # 0.5 are the element not classified as vassels but with the same feature of a vassel
            pro[lume_labels_new==i]=[0,0,255]
        elif row_values[i]==1: # tubuls
            pro[lume_labels_new==i]=[255,0,0]
        elif row_values[i]==3: # others elements ( glomeruli)
            pro[lume_labels_new==i]=[0,0,0]


    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(pro)
    plt.title('GT')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(lume_labels_new)
    plt.title('mask labels')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(labeled_image)
    plt.title('labels')
    plt.axis('off')
    plt.show()
