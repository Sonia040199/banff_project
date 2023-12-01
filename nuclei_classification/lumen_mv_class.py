r'''to classify the lumen using the majority voting applied to the classification of the nuclei
Autor Sonia'''
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage.morphology as sk
import cv2
import tifffile as tiff
import openpyxl
from bresenham import bresenham
from scipy import ndimage
import joblib

def contrast_augmentation(im, perc):
    '''     function to increse the contrast
    input:
    - im= image tomodify
    - perc= percentage of augmentatio. ( if perc=1 there is an augmentation of the 100%)
    output:
    - im_adjusted = modified image'''
    m = np.median(im)
    delta = perc * m
    im_adjusted = np.clip(im - delta, 0, 255).astype(np.uint8)
    return im_adjusted

def lumen_detecion(image,lume_size_up_th,lume_size_down_th,area):
    '''    function to obtain the lumen mask
    input:
    - image= image gray scale 
    - lume_size_up_th: max dimensione that a mask can have to be considered
    - lume_size_down_th: min dimensione that a mask can have to be considered
    - area: dimensione of the hole to remove
    output:
    - lumen_mask_fil = lumen mask after the post processing '''

    # PRE PROCESSING
    image_c=contrast_augmentation(image,1)

    # GLOBAL THRESHOLDING
    # inizialisation 
    lumen_mask = np.zeros_like(image_c, dtype=np.uint8)
    # threshold identification
    th, _= cv2.threshold(image_c, np.min(image_c), np.max(image_c), cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lumen_mask=np.where(image_c>=th,1,lumen_mask)
    lumen_mask_fn=lumen_mask.copy() 

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
        lumen_mask_fil[labels == largest_label] = 0 # check about the size of the bigger mask
    for i in range(num_labels):
        if stats[i, cv2.CC_STAT_AREA]<lume_size_down_th: # elimination of the mask with a size smaller than lume_size_down_th to clean the mask
            lumen_mask_fil[labels == i] = 0
        if stats[i, cv2.CC_STAT_AREA]>lume_size_up_th: # elimination of the mask with a size bigger than lume_size_up_th to remove the back ground
            lumen_mask_fil[labels == i] = 0

    # fill the small holes in the images
    lumen_mask_fil=sk.remove_small_holes(lumen_mask_fil,area)
    return(lumen_mask_fil)

def create_circular_kernel(radius):
    '''     fucntion to creata a circular kernel
    input:
    - radius of the circle
    output:
    -kernel '''
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    center = (radius, radius)
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            if distance <= radius:
                kernel[i, j] = 1
    return kernel
def num_nuclei_in_lume(nuclei_mask,lume_num_labels,lume_labels,kernel_big,r_g_c,for_lume_class):
    ''' function to know the number of nuclei arround each lumen.
    input:
    - nuclei_mask
    -lume_num_labels: number of labels in the lumen mask
    - lume_labels: labels for each element in the lumen mask
    - kernel_big: kernel used to identify the over extimate crown
    - r_g_c: channel of the red- channel of the green after a 100% increase of the contrast
    -for_lume_class: list of list in which the i-th list contain the labels of the nuclei connected to the i-th lumen of the image
    output:
    -for_lume_class: upgraited for_lume_class
    '''
    # is the same code of the previus fucntion but have to be executer before the feature extraction
    nuclei_num_labels=np.max(nuclei_mask)
    for i in range(nuclei_num_labels):
        if i==0:
            continue
        nuclei_i=np.zeros_like(nuclei_mask) 
        nuclei_i[nuclei_mask==i]=1
        radius=0
        intersection=[]
        kernel_circular=create_circular_kernel(1)
        for j in range(lume_num_labels):
            if j==0:
                continue
            lumen_i=np.zeros_like(image)
            lumen_i[lume_labels==j]=1
            dilatation_i=cv2.dilate(np.uint8(lumen_i),kernel_big)
            crown_i=dilatation_i-lumen_i
            intersection.append(np.sum(nuclei_i*crown_i))
        intersection=np.array(intersection)


        if len(np.where(intersection!=0)[0])==0: # if the nuclei is not intersecated with any crown is beacouse it is in the interstitia tissue
            pos=0
            for_lume_class[pos].append(i)


        elif len(np.where(intersection!=0)[0])==1:
            pos=int(np.where(intersection!=0)[0])+1
            for_lume_class[pos].append(i)

        elif len(np.where(intersection!=0)[0])>1:
    
            closer_lume_label=[]
    
            flag=0
            while flag==0:   
                radius+=1
                kernel_circular=create_circular_kernel(radius)
                dilatation_i=cv2.dilate(np.uint8(nuclei_i),np.uint8(kernel_circular))
                common_point=dilatation_i*lumen_mask
                if np.any(common_point!=0):
                    closer_lume_label_with_duplicate=lume_labels[np.where(common_point==1)]
                    closer_lume_label= list(set(closer_lume_label_with_duplicate))
                    if len(closer_lume_label)>1:
                        centroid_nuc = ndimage.measurements.center_of_mass(nuclei_i)
                        max_pixel_values=[]
                        for num in closer_lume_label:
                            pixel_values=[]
                            connected_lumen_i=np.where(lume_labels==num,1,0)
                            centroid_lum = ndimage.measurements.center_of_mass(connected_lumen_i)
                            point_line=list(bresenham(int(centroid_nuc[1]), int(centroid_nuc[0]),int(centroid_lum[1]), int(centroid_lum[0])))
                            for point in point_line:
                            
                                pixel_values.append(r_g_c[point[1],point[0]])             
                            max_pixel_values.append(np.max(pixel_values))
                        
                        if len(np.where(max_pixel_values==np.min(max_pixel_values))[0])>1:
                            conneceted_lume_label=closer_lume_label[int(np.where(max_pixel_values==np.min(max_pixel_values))[0][0])]
                            
                        else:
                            conneceted_lume_label=closer_lume_label[int(np.where(max_pixel_values==np.min(max_pixel_values))[0])]
                        
                        for_lume_class[conneceted_lume_label].append(i)
                        flag=1
                    else:
               
                      
                        for_lume_class[closer_lume_label[0]].append(i)
                        flag=1



    return(for_lume_class)

def classification(lumen_mask,image_rgb,y_pred,for_lume_class,with_crown,patch_name):
    ''' code to implement the majority voting
    input:
    -lumen_mask: mask of the lumen
    -image_rgb: RGB image of the patch
    -y_pred: prediction on the test set
    -for_lumen_class: is a list of list that contain in the i-th component all the nuclei connected to the i-th lumen
    -with_crown: list of list where the i-th element is refered to the i-th row of the feature matrix. each i-th list contain the label of the nuclei in the nuclei mask and the name of the patch
    -patch_name
    output:
    -output: RGB image with the results of the lumen classification'''
    output=image_rgb.copy()
    t=0
    v=0
    lume_num_labels, lume_labels, _, _ = cv2.connectedComponentsWithStats(np.uint8(lumen_mask))
    for i in range(lume_num_labels):
        t=0
        v=0
        if i==0:
            continue
        lumen_i=np.zeros_like(lumen_mask)
        lumen_i[lume_labels==i]=1
        connected_nu=for_lume_class[i]
        for pos,j in enumerate(with_crown):
            if j[1]==patch_name:

                if int(j[0]) in connected_nu:
                    if y_pred[pos]=='1.0':
                 
                        t+=1
                    else:
                  
                        v+=1
        if t>v:
             output[lumen_i==1]=[255,0,0]
        elif v>t:
            output[lumen_i==1]=[0,0,255]
        elif v==t:
            output[lumen_i==1]=[0,0,0]
        elif t & v:
            output[lumen_i==1]=[255,0,0]

    return(output)
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

    lume_size_down_th= 965.2666367999998
    lume_size_up_th=229638.144

    kernel_big=np.ones((90,90))
  
    parameters={'SVM':{},
               'KNN':{},
               'MLP':{},
               'RF':{}
               }

    set_folder_name=set_folder_test
    # load of the test set
    test=np.load(os.path.join(path,big_folder,set_folder_test,'test.npy'), allow_pickle=True)
    gt_test=np.load(os.path.join(path,big_folder,set_folder_test,'gt_test.npy'), allow_pickle=True)
    with_crown_test=np.load(os.path.join(path,big_folder,set_folder_test,'usable_test.npy'), allow_pickle=True)

    cont=0
    for patch_name in sorted(os.listdir(os.path.join(path,big_folder,set_folder_name,patch_folder))):
        if patch_name.split('.')[1]=='xml':
                continue

        matrice_prov=[]
        patch_name=patch_name.split('.')[0]


        image_rgb = np.array(Image.open(
                os.path.join(path,big_folder,set_folder_name,patch_folder,patch_name+'.jpg')
        ))

        image=np.array(Image.open(
                os.path.join(path,big_folder,set_folder_name,patch_folder,patch_name+'.jpg')
        ).convert('L'))

        nuclei_mask=tiff.imread(os.path.join(path,nuclei_mask_folder,patch_name+'_NucXav.tif'))

        # red channle- green channel
        r_g=abs(np.asarray(image_rgb)[:,:,0].astype(int)-np.asarray(image_rgb)[:,:,1].astype(int)).astype(np.uint8)
        r_g_c=contrast_augmentation(r_g,1).astype(np.uint8)


        # LUMEN DETECTION
        lumen_mask=lumen_detecion(np.asarray(image),lume_size_up_th,lume_size_down_th,120)
        lume_num_labels, lume_labels, _, _ = cv2.connectedComponentsWithStats(np.uint8(lumen_mask))
        
        
        for_lume_class=[]
    
        for i in range(lume_num_labels):
            for_lume_class.append([])

        for_lume_class=num_nuclei_in_lume(nuclei_mask,lume_num_labels,lume_labels,kernel_big,r_g_c,for_lume_class)


        # CLASSIFICATION OF THE NUCLEI
        print('nuclei calssification')


        plt.figure()
        for kk,k in enumerate(parameters.keys()):
            model= joblib.load(os.path.join(os.path.join(path,folder_model,k+'.pkl'))) 
            y_pred=model.predict(test)
            # CLASSIFICATION OF THE LUMEN
            print('lumen calssification')
            final1=classification(lumen_mask,image_rgb,y_pred,for_lume_class,with_crown_test,patch_name)
            plt.subplot(2,3,1)
            plt.imshow(image_rgb)
            plt.title('original')
            plt.axis('off')
            plt.subplot(2,3,kk+2)
            plt.imshow(final1)
            plt.title(k)
            plt.axis('off')
        plt.show()

 

