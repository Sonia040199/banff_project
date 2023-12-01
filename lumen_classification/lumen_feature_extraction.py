r''' to implement the feature extraction on new images without the ground thruth
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
from sklearn.preprocessing import MinMaxScaler

def contrast_augmentation(im, perc):
    '''     function to increse the contrast
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
    '''    function to obtain the lumen mask
    input:
    - image= image gray scale 
    - lume_size_up_th: max dimensione that a mask can have to be considered
    - lume_size_down_th: min dimensione that a mask can have to be considered
    - area: dimensione of the hole to remove
    output:
    - lumen_mask_fil = lumen mask after the post processing '''


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
    '''     function to connect the mask from the same lumen
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


def assosation_nuclei_lumes(nuclei_mask,lume_labels_new,lume_num_label_new,r_g_c):
    ''' fucntion to associate the nuclei to the lumen mask
input: 
- nuclei_mask: mask of the nuclei
- lumen_labels_new: labels of the mask of the lumen after the corrections
- lumen_num_label_new: max label in lume_labels_new
- r_g_c: channel red- channel green with contrast augmentation
output:
nuclei_in_lumen: the element j-th of the vectore nuclei_in_lumen contain a list with the values of the nuclei connected to the j-th lumen of the mask 
conteggio_nuclei_lume: the element j-th of the vectore conteggio_nuclei_lume contain the number of nuclei connectes to the j-th lumen of the mask '''

    nuclei_num_labels=np.max(nuclei_mask) # exctraction of the max label in nuclei_mask
    #variable initialisation
    nuclei_in_lumen=[]
    conteggio_nuclei_lume=np.zeros(lume_num_label_new)
    nuclei_in_lumen.append([])

    for i in range(nuclei_num_labels): # analysis of each nuclei
        if i==0:
            continue
        nuclei_i=np.zeros_like(nuclei_mask) # mask of the single nuclei
        nuclei_i[nuclei_mask==i]=1
        intersection=[]
       
        for j in range(lume_num_label_new):
            if j==0:
                continue
            lumen_i=np.zeros_like(image)
            lumen_i[lume_labels_new==j]=1
            nuclei_in_lumen.append([]) # i have a list for each nuclei
            # crown evaluation
            dilatation_i=cv2.dilate(np.uint8(lumen_i),kernel_big)
            crown_i=dilatation_i-lumen_i

            intersection.append(np.sum(nuclei_i*crown_i))
        intersection=np.array(intersection) # intersection has inside the values of the intersection between each corn and the considered nuclei

        if len(np.where(intersection!=0)[0])==0: # if the nuclei is not intersecated with any crown is beacouse it is in the interstitia tissue
            pos=0

        elif len(np.where(intersection!=0)[0])==1: # if it is connected to a single crown i associate the nuclei to the lumen that generates the crown
            pos=int(np.where(intersection!=0)[0])+1
    
        elif len(np.where(intersection!=0)[0])>1: # if it is in more crown   
            closer_lume_label=np.where(intersection!=0)[0]
            closer_lume_label+=1
            centroid_nuc = ndimage.measurements.center_of_mass(nuclei_i) # evaluation of the centroid of each nuclei mask
            max_pixel_values=[]
            for num in closer_lume_label:
                pixel_values=[]
                connected_lumen_i=np.where(lume_labels_new==num,1,0) # evaluation of the label of the lumen to whcihc the nuclei is connected
                centroid_lum = ndimage.measurements.center_of_mass(connected_lumen_i) # evaluation of the centroid for each connecte lumen
                point_line=list(bresenham(int(centroid_nuc[1]), int(centroid_nuc[0]),int(centroid_lum[1]), int(centroid_lum[0]))) # bresenham distance between the centroids
                for point in point_line:
                    pixel_values.append(r_g_c[point[1],point[0]])
                max_pixel_values.append(np.max(pixel_values)) # max_pixel_values has inside the max values of the pixel of the bresenham path considering the r_g_c image
                if len(np.where(max_pixel_values==np.min(max_pixel_values))[0])>1:
                        conneceted_lume_label=closer_lume_label[int(np.where(max_pixel_values==np.min(max_pixel_values))[0][0])] # if there are more lumen that has the same min of the max i take the first
                else:
                        conneceted_lume_label=closer_lume_label[int(np.where(max_pixel_values==np.min(max_pixel_values))[0])] # connect the nuclei to the lumen that give the minum values of the max
                pos=conneceted_lume_label

        nuclei_in_lumen[pos].append(i) 
        conteggio_nuclei_lume[pos]+=1
    return(nuclei_in_lumen,conteggio_nuclei_lume)


def excel_extractor(path,excel_file,patch_name):
    '''     to extract the labels from the excel file
    input: 
    path: path of the folder in which there is the cose
    main_folder: folder of the excel file
    excel_file: name of the excel file
    patch_name
    output:
    gt_tot: vectore with the ground thruth'''


    # exctraction of each line of the excel file
    excel=openpyxl.load_workbook(os.path.join(path,excel_file+'.xlsx'))
    sheet=excel.get_sheet_by_name('Sheet1')
    for i in range(sheet.max_row):
        if not isinstance(sheet[i+1][0].value,str):
            continue
        row_values = []
        gt_tot=[]
        for cell in sheet[i+1]:
            row_values.append(cell.value)

        patch_name_ex=row_values[0]
    
        if patch_name_ex==patch_name: # analysis of the line related to the considered image
            row_values.pop(0)
        
            for item in row_values:
                if item==0.5:
                    gt_tot.append(0) 
                elif item==1 or item==0 or item==2 or item==3:
                    gt_tot.append(item)

            break

    return(gt_tot)



def color_features(image,mask,feature,feature_name,name,mask_kind):
    '''    to exctract the color feature
    input:
    image: image from which extract the color feature
    mask: the function evaluate the feature just in the part of the image in which the mask is white
    feature: matrix of the feature to upgraide
    feature_name: vector with the name of the feature to upgraid
    name: name of the kind of image 
    mask_kind: name of the kind of mask
    output:
    upgraidted feature and feature_name
 '''

    #average of pixel values
    media=np.mean(image[np.where(mask==1)])
    feature.append(media)
    feature_name.append(f'mean_'+name+'_'+mask_kind)


    #median of the pixel values
    median=np.median(image[np.where(mask==1)])
    feature.append(median)
    feature_name.append(f'median_'+name+'_'+mask_kind)

    #std of the pixel values
    std=np.std(image[np.where(mask==1)])
    feature_name.append(f'std_'+name+'_'+mask_kind)
    feature.append(std)
    return(feature,feature_name)

def color_features_nuclei(image,mask,dic,name):
    '''     to extract the color feature related to the nuclei
    input:
    image: image from which extract the color feature
    mask: the function evaluate the feature just in the part of the image in which the mask is white
    dic: dictionary of each nuclei
    name: name of the image kind
    output:
    upgraidet dictionary'''


    media=np.mean(image[np.where(mask==1)])
    dic[f'mean_{name}_nuclei'].append(media)

    median=np.median(image[np.where(mask==1)])
    dic[f'median_{name}_nuclei'].append(median)

    std=np.std(image[np.where(mask==1)])
    dic[f'std_{name}_nuclei'].append(std)
    return(dic)

def shape_features(mask):
    ''' fucntion to evaluete the eccentricity, the height and width of the mask given as input '''
 
    contours, _= cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected_contourn=contours[0]
   # eccentricitá
    ellipse = cv2.fitEllipse(selected_contourn)
    eccentricity = np.sqrt(1 - (ellipse[1][0] / ellipse[1][1])**2)
    min_axes=ellipse[1][1] # width
    max_axes=ellipse[1][0] # height

    return(eccentricity,min_axes,max_axes)

def convex_hull(mask):
    ''' fucntion to extract the convex hull from the mask'''
 
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hul_enelope_mask=np.zeros_like(mask)
    contour_points = []
    for contour in contours:
        for point in contour:
            contour_points.append([point[0][1],point[0][0]])
    contour_points = np.vstack(contour_points)
    
    hull = ConvexHull(contour_points[:, ::-1])
    vertices_indices = hull.vertices
    cv2.drawContours(hul_enelope_mask, [contour_points[vertices_indices][:, ::-1]], -1, (255, 255, 255), thickness=cv2.FILLED)
    difference=hul_enelope_mask.copy()
    difference[mask==1]=0

    return(np.sum(difference)/np.sum(mask)) # output= difference between the convex hull and the mask


def feature_extractor(lumen_i,nuclei_mask,number_nuclei,nuclei_in_lumen,kernel_crown,r,g,b,image,feature,feature_name,i):
    '''     input:
    -lumen_i: mask of the i-th lumen
    -nuclei_mask: nuclei mask
    -number_nuclei: secondo output of assosation_nuclei_lumes
    -nuclei_in_lumen: first output of assosation_nuclei_lumes
    -kernel_crown: kernel used to obtain the mask of the krown
    -r: red channel of the RGB image
    -g: green channel of the RGB image
    -b: blue channel of the RGB image
    -image: gray scale image
    -feature: matrix for the feature
    -feature_name: vector for the name of the feature
    -i: label of the lumen mask
    output:
    -feature: matrix for the feature
    -feature_name: vector for the name of the feature'''


    # f1--> size of the lumen
    feature.append(np.sum(lumen_i))
    feature_name.append('lumen_size')
    # f2--f4> shape feature
    eccentr,width,heigth=shape_features(lumen_i)
    feature.append(width)
    feature_name.append('width_lumen')
    feature.append(heigth)
    feature_name.append('hight_lumen')
    feature.append(eccentr)
    feature_name.append('lumen_eccentricity')

    # f5--> convex hull
    feature.append(convex_hull(lumen_i))
    feature_name.append('convex_hull')

    # f6--> number of nuclei connected
    feature.append(number_nuclei[i])
    feature_name.append('number_nuclei')

    # f7--f18 --> color feature crown
    dilatate_lumen=cv2.dilate(np.uint8(lumen_i),kernel_crown)
    crown=dilatate_lumen-lumen_i
    crown[nuclei_mask>0]=0

    feature,feature_name=color_features(r,crown,feature,feature_name,'r','crown')
    feature,feature_name=color_features(g,crown,feature,feature_name,'g','crown')
    feature,feature_name=color_features(b,crown,feature,feature_name,'b','crown')
    feature,feature_name=color_features(image,crown,feature,feature_name,'gr','crown')

    # --> feature of the nuclei arround the lumen
    nuclei_feature={'nuclei_size':[],
                    'nuclei_shape':[],
                    'mean_r_nuclei':[],
                    'median_r_nuclei':[],
                    'std_r_nuclei':[],
                    'mean_g_nuclei':[],
                    'median_g_nuclei':[],
                    'std_g_nuclei':[],
                    'mean_b_nuclei':[],
                    'median_b_nuclei':[],
                    'std_b_nuclei':[],
                    'mean_gr_nuclei':[],
                    'median_gr_nuclei':[],
                    'std_gr_nuclei':[]}
    

    if len(nuclei_in_lumen[i])==0:
        for key in nuclei_feature.keys():
            nuclei_feature[key]=0
    else:
        for n in nuclei_in_lumen[i]:
            nucleo_i=np.where(nuclei_mask==n,1,0)
            # f19--20 --> mean size 
            nuclei_feature['nuclei_size'].append(np.sum(nucleo_i))
            # f20--21 --> mean shape
            nuclei_feature['nuclei_shape'].append(shape_features(nucleo_i))

            # f22--32 --> mean color feature
            nuclei_feature=color_features_nuclei(r,nucleo_i,nuclei_feature,'r')
            nuclei_feature=color_features_nuclei(g,nucleo_i,nuclei_feature,'g')
            nuclei_feature=color_features_nuclei(b,nucleo_i,nuclei_feature,'b')
            nuclei_feature=color_features_nuclei(image,nucleo_i,nuclei_feature,'gr')

    for key in nuclei_feature.keys():
        feature.append(np.mean(nuclei_feature[key]))
        feature_name.append(f'mean_{key}')

        feature.append(np.std(nuclei_feature[key]))
        feature_name.append(f'std_{key}')

    return(feature,feature_name)

def data_set_division(gt,matrice,usable):
    '''     function for balance the classes in the train set:
    input:
    - gt: ground thrut
    - matrice: feature matrix
    - usable: usable lumen
    output:
    train: feature matrix balances
    gt and usable related to the nuclei in the train'''

    matrice_1=[] # for nuclei
    matrice_0=[] # for vassels
    # I divide the data according to the calsse by hanging the class itself at the bottom
    for i, lb in enumerate(gt):
        v=matrice[i,:]
        if lb==1:
            v=np.append(v,1)
            v=np.append(v,usable[i])
            matrice_1.append(v)
        else:
            v=np.append(v,0)
            v=np.append(v,usable[i])
            matrice_0.append(v)
   # I obtain in this way two matrices one for calsse 0 and one for calsse 1
    matrice_1 = np.vstack(matrice_1)
    matrice_0 = np.vstack(matrice_0)

    if matrice_1.shape[0]> matrice_0.shape[0]:
        num=matrice_0.shape[0]
    else:
        num=matrice_1.shape[0]

    train_1 = matrice_1[:num, :]

    train_0 = matrice_0[:num, :]
    train_plus_gt = np.vstack((train_1, train_0))
    np.random.shuffle(train_plus_gt)
    gt_train=train_plus_gt[:,-3]
    usable_train=train_plus_gt[:,-2:]
    train=np.delete(train_plus_gt,np.s_[-3:], axis=1)
    return(train,gt_train,usable_train)

def cambio_gt(gt):
    ''' fucntion to trasform the gt in string following the output of the models'''
   
    gt_new=[]
    for i in gt:
        if i==1:
            gt_new.append('1.0')
        else:
            gt_new.append('0.0')
    return(np.array(gt_new))

if __name__=='__main__':

    # folders names
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



    #constant definition
    lume_size_down_th= 965.2666367999998/2
    lume_size_up_th=229638.144
    area_holes=120
    kernel_big=np.ones((90,90))
    kernel_crown=np.ones((40,40))
    matrice=[]
    gt=[]
    usable=[]
    barra=0
    contatore_lumi=0


    # to chose the kind of data set 
    set_folder=set_folder_bad_image


    for patch_name in sorted(os.listdir(os.path.join(path,big_folder,set_folder,patch_folder))):
        patch_name=patch_name.split('.')[0]
        barra+=1
        print(f'-----------------------processing: {patch_name}        {barra}/{len(os.listdir(os.path.join(path,big_folder,set_folder,patch_folder)))}-----------------------')

        # #load of the original image
        #RGB image
        image_rgb = np.array(Image.open(
                    os.path.join(path,big_folder,set_folder,patch_folder,patch_name+'.jpg')
            ))
        #gray scale image
        image=np.array(Image.open(
                    os.path.join(path,big_folder,set_folder,patch_folder,patch_name+'.jpg')
            ).convert('L'))
        
        #-----------------------------------------------------------just when set_folder=set_folder_fake_pas--------------------------------------------------------------------------------
        # # RGB image
        # image_rgb = np.array(Image.open(
        #             os.path.join(path,big_folder,set_folder,patch_folder,patch_name+'.png')
        #     ))
        
        # #gray scale image
        # image=np.array(Image.open(
        #             os.path.join(path,big_folder,set_folder,patch_folder,patch_name+'.png')
        #     ).convert('L'))
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        print('SEGMENTATION')

        # LUMEN DETECTION with threshold
        lumen_mask_th=lumen_detecion_thresholding(np.asarray(image),lume_size_up_th,lume_size_down_th,area_holes)
        
        # channel division e combination
        r=np.asarray(image_rgb)[:,:,0]
        g=np.asarray(image_rgb)[:,:,1]
        b=np.asarray(image_rgb)[:,:,2]
        r_g=abs(np.asarray(image_rgb)[:,:,0].astype(int)-np.asarray(image_rgb)[:,:,1].astype(int)).astype(np.uint8)
        r_g_c=contrast_augmentation(r_g,1).astype(np.uint8)

        # improving lumen mask with the aggregation of the smaller mask that belong to the same lumen
        lume_labels_new=mask_aggregation(r_g_c,lumen_mask_th)
        lume_num_label_new=np.max(lume_labels_new)+1
        contatore_lumi+=lume_num_label_new
        lumen_mask=np.where(lume_labels_new!=0,1,0)

        # NUCLEI DETECTION
        nuclei_mask=tiff.imread(os.path.join(path,nuclei_mask_folder,patch_name+'_NucXav.tif')) #load of the mask
        #nuclei mask is not a binary mask but contain the values of the mask of each nuclei. Like lumen_labels
        nuclei_num_labels=np.max(nuclei_mask)
        print('done')


        # GT extraction and assosation nuclei to lumen
        gt_tot=excel_extractor(path,excel_file,patch_name)
        nuclei_in_lumen,number_nuclei=assosation_nuclei_lumes(nuclei_mask,lume_labels_new,lume_num_label_new,r_g_c)

        print('FEATURE EXTRACTION')

        for i in range(lume_num_label_new):
            if i==0:
                continue
            if gt_tot[i-1]==2 or gt_tot[i-1]==3: # for the training i will not considering the error masks and the glomerulai
                continue

            lumen_i=np.zeros_like(image)
            lumen_i[lume_labels_new==i]=1
            gt.append(gt_tot[i-1])
            usable.append([i,patch_name]) # usable is a list of list in which each element has inside the value of the lumen and the name of the patch from which cames from
            feature=[]
            feature_name=[]
            feature,feature_name=feature_extractor(lumen_i,nuclei_mask,number_nuclei,nuclei_in_lumen,kernel_crown,r,g,b,image,feature,feature_name,i)
            matrice.append(feature) # for each patch
    
    matrice = np.vstack(matrice) # creation of the 2D matrix

    print('done')
   
    if set_folder=='TRAIN':
        print('train balancing and normalisation')
        # balancing
        train_not_norm,gt_train,usable_train=data_set_division(gt,matrice,usable)
        # min max scaling normalisation
        scaler_train = MinMaxScaler()
        train=scaler_train.fit_transform(train_not_norm)
        np.save(os.path.join(path,big_folder,set_folder,'train.npy'), train)
        np.save(os.path.join(path,big_folder,set_folder,'gt_train.npy'), gt_train)
        np.save(os.path.join(path,big_folder,set_folder,'usable_train.npy'), usable_train)
        np.save(os.path.join(path,big_folder,set_folder,'feature_name.npy'), feature_name)
        print('saved')
    elif set_folder=='VALIDATION':
        print('normalisation')
        # min max scaling normalisation
        scaler_val = MinMaxScaler()
        validation=scaler_val.fit_transform(matrice)
        # validation=validation[:,0]
        gt_val=cambio_gt(gt)
        np.save(os.path.join(path,big_folder,set_folder,'validation.npy'), validation)
        np.save(os.path.join(path,big_folder,set_folder,'gt_val.npy'), gt_val)
        np.save(os.path.join(path,big_folder,set_folder,'usable_val.npy'), usable)
        print('saved')
    elif set_folder=='TEST':
        print('normalisation')
        # min max scaling normalisation
        scaler_test = MinMaxScaler()
        test=scaler_test.fit_transform(matrice)
        gt_test=cambio_gt(gt)

        np.save(os.path.join(path,big_folder,set_folder,'test.npy'), test)
        np.save(os.path.join(path,big_folder,set_folder,'gt_test.npy'), gt_test)
        np.save(os.path.join(path,big_folder,set_folder,'usable_test.npy'), usable)
        print('saved')
    else:
        print('normalisation')
        # min max scaling normalisation
        scaler = MinMaxScaler()
        matrice_norm=scaler.fit_transform(matrice)
        gt_str=cambio_gt(gt)

        np.save(os.path.join(path,big_folder,set_folder,'matrice.npy'), matrice_norm)
        np.save(os.path.join(path,big_folder,set_folder,'gt.npy'), gt_str)
        np.save(os.path.join(path,big_folder,set_folder,'usable.npy'), usable)
        np.save(os.path.join(path,big_folder,set_folder,'feature_name.npy'), feature_name)
        print('saved')

  
    print(f'-number of total lumen: {contatore_lumi}\n-number of utilised lumen: {matrice.shape[0]}\n-number of feature: {matrice.shape[1]}')