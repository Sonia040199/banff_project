r'''for the feature extraction of the nuclei. It saved the matrix of the feature extraction, the nuclei included in a crown and the grpund thruth in the set folder
Autor Sonia '''
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
from scipy.spatial import ConvexHull
from sklearn.preprocessing import MinMaxScaler


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

def connected_lumen(nuclei_i,lume_num_labels,lume_labels,kernel_big,r_g_c,lumen_mask):
    '''fucntion to find the lumen to which the i-th nuclei is connected
    input:
    - nuclei_i: mask of the i-th nuclei
    - lume_num_labels: number of labels in the lumen mask
    - lume_labels: labels for each element in the lumen mask
    - kernel_big: kernel used to identify the over extimate crown
    - r_g_c: channel of the red- channel of the green after a 100% increase of the contrast
    - lumen_mask
    output:
    - crown flag: variable to know if the i-th nuclei is insered in a flag or not. if crown_flag==0 the i-th nuclei will not be considered
    -connected_lumen_mask: mask of the connected lumen
    -pos: label of the connected lumen mask
    '''
    #kernel creation
    radius=0
    intersection=[]
    kernel_circular=create_circular_kernel(1)
    for j in range(lume_num_labels):
        if j==0:
            continue
        lumen_i=np.zeros_like(image)
        lumen_i[lume_labels==j]=1
        dilatation_i=cv2.dilate(np.uint8(lumen_i),kernel_big)
        crown_i=dilatation_i-lumen_i #crown mask creation
        intersection.append(np.sum(nuclei_i*crown_i)) #evaluation of the intersection between the nuclei mask and the crown mask
    intersection=np.array(intersection)

    if len(np.where(intersection!=0)[0])==0: # if the nuclei is not intersecated with any crown is beacouse it is in the interstitia tissue
        crown_flag=0
        connected_lumen_mask=np.zeros_like(nuclei_i)
        pos=0
    elif len(np.where(intersection!=0)[0])==1: # if it is connected to a single crown i associate the nuclei to the lumen that generates the crown
        connected_lumen_mask=np.where(lume_labels==(int(np.where(intersection!=0)[0])+1),1,0)
        crown_flag=1
        pos=int(np.where(intersection!=0)[0])+1

    elif len(np.where(intersection!=0)[0])>1: # if it is in more crown  
        closer_lume_label=[]
        crown_flag=1
        flag=0
        while flag==0: # in the while cicle i dilatate the nuclei mask untill it tuch the mask of a lumen
            radius+=1
            kernel_circular=create_circular_kernel(radius) # every while iteratio i increase the radius of the kernel
            dilatation_i=cv2.dilate(np.uint8(nuclei_i),np.uint8(kernel_circular))
            common_point=dilatation_i*lumen_mask
            if np.any(common_point!=0): # if the expanded mask tuch one  lumen mask
                # list of lumen mask label for which i have a contact with the nuclei mask
                closer_lume_label_with_duplicate=lume_labels[np.where(common_point==1)] 
                closer_lume_label= list(set(closer_lume_label_with_duplicate))
                if len(closer_lume_label)>1: # if the nuclei tuch more than one lumen at the same time i evaluate the breshenam path
                    centroid_nuc = ndimage.measurements.center_of_mass(nuclei_i) # centroid of the nuclei
                    max_pixel_values=[]
                    for num in closer_lume_label:
                        pixel_values=[]
                        connected_lumen_i=np.where(lume_labels==num,1,0)
                        centroid_lum = ndimage.measurements.center_of_mass(connected_lumen_i) # centroid of the lumen
                        point_line=list(bresenham(int(centroid_nuc[1]), int(centroid_nuc[0]),int(centroid_lum[1]), int(centroid_lum[0]))) # bresenham path between the two centroid
                        for point in point_line:
                            pixel_values.append(r_g_c[point[1],point[0]])
                        max_pixel_values.append(np.max(pixel_values))

                    if len(np.where(max_pixel_values==np.min(max_pixel_values))[0])>1:
                            conneceted_lume_label=closer_lume_label[int(np.where(max_pixel_values==np.min(max_pixel_values))[0][0])] # if there are more lumen that has the same min of the max i take the first
                    else:
                            conneceted_lume_label=closer_lume_label[int(np.where(max_pixel_values==np.min(max_pixel_values))[0])] # connect the nuclei to the lumen that give the minum values of the max
                    
                    pos=conneceted_lume_label
                    connected_lumen_mask=np.where(lume_labels==conneceted_lume_label,1,0)
                    flag=1
                else:
                    connected_lumen_mask=np.where(lume_labels==closer_lume_label,1,0)
                    pos=closer_lume_label
                    flag=1
    return(crown_flag,connected_lumen_mask,pos)

def num_nuclei_in_lume(nuclei_mask,lume_num_labels,lume_labels,kernel_big,r_g_c,number_nuclei_in_lumen):
    ''' function to know the number of nuclei arround each lumen.
    input:
    - nuclei_mask
    -lume_num_labels: number of labels in the lumen mask
    - lume_labels: labels for each element in the lumen mask
    - kernel_big: kernel used to identify the over extimate crown
    - r_g_c: channel of the red- channel of the green after a 100% increase of the contrast
    -number_nuclei_in_lumen: number of nuclei arrounf each lumen
    output:
    -number_nuclei_in_lumen: number of nuclei arrounf each lumen
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

        if len(np.where(intersection!=0)[0])==1:
            pos=int(np.where(intersection!=0)[0])+1
            number_nuclei_in_lumen[pos]+=1



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
                        
                        number_nuclei_in_lumen[conneceted_lume_label]+=1
                        flag=1
                    else:
                        number_nuclei_in_lumen[closer_lume_label]+=1
                        flag=1



    return(number_nuclei_in_lumen)


def color_features(image,mask,feature,feature_name,name):
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
    feature_name.append(f'media_'+name)

    #average of pixel values
    median=np.median(image[np.where(mask==1)])
    feature.append(median)
    feature_name.append(f'median_'+name)

    #std of the pixel values
    std=np.std(image[np.where(mask==1)])
    feature_name.append(f'std_'+name)
    feature.append(std)
    return(feature,feature_name)

def shape_features(mask):
    ''' fucntion to evaluete the eccentricity, and the hull moment of mask given as input '''
    contours, _= cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected_contourn=contours[0]
   # eccentricitá
    ellipse = cv2.fitEllipse(selected_contourn)
    eccentricity = np.sqrt(1 - (ellipse[1][0] / ellipse[1][1])**2)
    # hull moment
    Mo = cv2.moments(selected_contourn)
    hu_moments = cv2.HuMoments(Mo)
    return(eccentricity,hu_moments)

def hull_envelope(mask):
    ''' fucntion to evaluete the eccentricity, the height and width of the mask given as input '''
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
    return(np.sum(difference)) # output= difference between the convex hull and the mask

def feature_extraction(feature,feature_name,mask_connected_lumen,nuclei_i,r,g,b,image,number_nuclei_in_lumen,pos):
   
    '''     input:
    -feature: matrix for the feature
    -feature_name: vector for the name of the feature
    -mask_connected_lumen: mask of the connected lumen
    -nuclei_i: masck of the i-th nuclei
    -r: red channel of the RGB image
    -g: green channel of the RGB image
    -b: blue channel of the RGB image
    -image: gray scale image
    -number_nuclei_in_lumen: number of nuclei arround each lumen
    -pos: label of the connected nuclei
     output:
    -feature: matrix for the feature
    -feature_name: vector for the name of the feature

 '''
    # feature N_1: connected lumen dimension
    feature.append(np.sum(mask_connected_lumen))
    feature_name.append('lumen_size')

    # feature N_2: nuclei dimension
    feature.append(np.sum(nuclei_i))
    feature_name.append('nucleus_size')

    # feature N_3-N_14: nuclei colour (I take mean, median and standard deviation of each channel of the RGB image and the grey scale image so you have 3x4=12 features for core colour)
    feature,feature_name=color_features(r,nuclei_i,feature,feature_name,'r')
    feature,feature_name=color_features(g,nuclei_i,feature,feature_name,'g')
    feature,feature_name=color_features(b,nuclei_i,feature,feature_name,'b')
    feature,feature_name=color_features(image,nuclei_i,feature,feature_name,'gr')

    # feature N_15-N_22: lumen shape (you have 7 hull moments and eccentricity so 8 features)
    eccentricity,hu_moments=shape_features(mask_connected_lumen)
    feature.append(eccentricity)
    feature_name.append('eccenticity_lumen')
    for k in range(len(hu_moments)):
        feature.append(hu_moments[k])
        feature_name.append(f'hu_{k+1}_lumen')
    # feature N_23-N_30: nuclei shape (you have 7 hull moments and eccentricity so 8 features)
    eccentricity,hu_moments=shape_features(nuclei_i)
    feature.append(eccentricity)
    feature_name.append('eccenticity_nucleus')
    for k in range(len(hu_moments)):
        feature.append(hu_moments[k])
        feature_name.append(f'hu_{k+1}_nucleus')

    # feature N_31: nnumber of nuclei around the same lumen
    feature.append(number_nuclei_in_lumen[pos])
    feature_name.append(f'n_nuclei')

    # feature N_32: convex hull
    he=hull_envelope(mask_connected_lumen)
    feature.append(he)
    feature_name.append(f'hull_enevelope')
    return(feature,feature_name)


def data_set_division(gt,matrice,with_crown):
    '''     function for balance the classes in the train set:
    input:
    - gt: ground thrut
    - matrice: feature matrix
    - with_crown: usable nuclei
    output:
    train: feature matrix balances
    gt and with_crown related to the nuclei in the train'''
    matrice_1=[] # for nuclei
    matrice_0=[] # for vassels
    # I divide the data according to the calsse by hanging the class itself at the bottom
    for i, lb in enumerate(gt):
        v=matrice[i,:]
        if lb==1:
            v=np.append(v,1)
            v=np.append(v,with_crown[i])
            matrice_1.append(v)
        else:
            v=np.append(v,0)
            v=np.append(v,with_crown[i])
            matrice_0.append(v)
    # I obtain in this way two matrices one for calsse 0 and one for calsse 1
    matrice_1 = np.vstack(matrice_1)
    np.random.shuffle(matrice_1)
    matrice_0 = np.vstack(matrice_0)
    np.random.shuffle(matrice_0)
    if matrice_1.shape[0]> matrice_0.shape[0]:
        num=matrice_0.shape[0]
    else:
        num=matrice_1.shape[0]

    train_1 = matrice_1[:num, :]

    train_0 = matrice_0[:num, :]
    train_plus_gt = np.vstack((train_1, train_0))
    gt_train=train_plus_gt[:,-3]
    with_crown_train=train_plus_gt[:,-2:]
    train=np.delete(train_plus_gt,np.s_[-3:], axis=1)
    return(train,gt_train,with_crown_train)



if __name__=='__main__':
    # folders names
    path = os.getcwd()
    big_folder='DATA_SET'
    set_folder_train='TRAIN'
    set_folder_test='TEST'
    patch_folder='patches'
    nuclei_mask_folder='nuclei_mask'
    excel_file='nuclei_lables'

    lume_size_down_th= 965.2666367999998/2
    lume_size_up_th=229638.144
    kernel_big=np.ones((90,90))
    matrice=[]
    gt=[]
    with_crown=[]
    barra=0
    contatore_utili=0

    # to chose the kind of data set 
    set_folder=set_folder_train


    for patch_name in sorted(os.listdir(os.path.join(path,big_folder,set_folder,patch_folder))):

        patch_name=patch_name.split('.')[0]
    # patch_name='image_x115712_y29696_1'
        barra+=1
        print(f'-----------------------processing: {patch_name}        {barra}/{len(os.listdir(os.path.join(path,big_folder,set_folder,patch_folder)))}-----------------------')

        #RGB image
        image_rgb = np.array(Image.open(
                os.path.join(path,big_folder,set_folder,patch_folder,patch_name+'.jpg')
        ))
        #gray scale image
        image=np.array(Image.open(
                os.path.join(path,big_folder,set_folder,patch_folder,patch_name+'.jpg')
        ).convert('L'))

        # red channel
        r=np.asarray(image_rgb)[:,:,0]
        # green channel
        g=np.asarray(image_rgb)[:,:,1]
        #blue channel
        b=np.asarray(image_rgb)[:,:,2]
        # red channle- green channel
        r_g=abs(np.asarray(image_rgb)[:,:,0].astype(int)-np.asarray(image_rgb)[:,:,1].astype(int)).astype(np.uint8)
        r_g_c=contrast_augmentation(r_g,1).astype(np.uint8)


        # LUMEN DETECTION
        lumen_mask=lumen_detecion(np.asarray(image),lume_size_up_th,lume_size_down_th,120)
        lume_num_labels, lume_labels, _, _ = cv2.connectedComponentsWithStats(np.uint8(lumen_mask))
        number_nuclei_in_lumen=np.zeros(lume_num_labels)

        # NUCLEI DETECTION
        nuclei_mask=tiff.imread(os.path.join(path,nuclei_mask_folder,patch_name+'_NucXav.tif'))
        nuclei_num_labels=np.max(nuclei_mask)


        # GT extraction
        excel=openpyxl.load_workbook(os.path.join(path,big_folder,set_folder,excel_file+'.xlsx'))
        #extraction of the row of the excel file
        sheet=excel.get_sheet_by_name('Sheet1')
        for i in range(sheet.max_row):
            if not isinstance(sheet[i+1][0].value,str):
                continue
            row_values = []
            gt_tot=[]
            for cell in sheet[i+1]:
                row_values.append(cell.value)

            patch_name_ex=row_values[0]
            if patch_name_ex==patch_name:
                row_values.pop(0)
            
                for j,item in enumerate(row_values):
                    if item==0.5:
                        gt_tot.append(0)
                    elif item==1.5:
                        gt_tot.append(1)
                    elif item==1 or item==0 or item==2:
                        gt_tot.append(item)

                break



        # count number of nuclei around each lumen

        number_nuclei_in_lumen=num_nuclei_in_lume(nuclei_mask,lume_num_labels,lume_labels,kernel_big,r_g_c,number_nuclei_in_lumen)

        
    #FEATURE EXTRACTION 
        print('feature extraction')
        for i in range(nuclei_num_labels):
            if i==0:
                continue
            nuclei_i=np.zeros_like(nuclei_mask) # mask of the i-th nuclei
            nuclei_i[nuclei_mask==i]=1
            feature=[]
            feature_name=[]

            # find the connected lumen
            crown_flag,mask_connected_lumen,pos=connected_lumen(nuclei_i,lume_num_labels,lume_labels,kernel_big,r_g_c,lumen_mask)
    

            if crown_flag==0: # it mean that the nucleus that i'm analysing is not connected to any crown
                continue
            if gt_tot[i-1]==2:
                continue
            contatore_utili+=1
            
            # I keep track of the nuclei whose features I am extracting and their class
            gt.append(gt_tot[i-1])
            with_crown.append([i,patch_name])

            feature,feature_name=feature_extraction(feature,feature_name,mask_connected_lumen,nuclei_i,r,g,b,image,number_nuclei_in_lumen,pos)

            matrice.append(feature)
        print('done')

    matrice = np.vstack(matrice)



    if set_folder=='TRAIN':
        print('train balancing and normalisation')
        # balancing
        train_not_norm,gt_train,with_crown_train=data_set_division(gt,matrice,with_crown)
        # min max scaling normalisation
        scaler_train = MinMaxScaler()
        train=scaler_train.fit_transform(train_not_norm)
        np.save(os.path.join(path,big_folder,set_folder,'train.npy'), train)
        np.save(os.path.join(path,big_folder,set_folder,'gt_train.npy'), gt_train)
        np.save(os.path.join(path,big_folder,set_folder,'usable_train.npy'), with_crown)
        np.save(os.path.join(path,big_folder,set_folder,'feature_name.npy'), feature_name)
        print('saved')
    elif set_folder=='TEST':
        print('normalisation')
        # min max scaling normalisation
        scaler_test = MinMaxScaler()
        test=scaler_test.fit_transform(matrice)
        gt_test=cambio_gt(gt)
        np.save(os.path.join(path,big_folder,set_folder,'test.npy'), test)
        np.save(os.path.join(path,big_folder,set_folder,'gt_test.npy'), gt_test)
        np.save(os.path.join(path,big_folder,set_folder,'usable_test.npy'), with_crown)
        print('saved')
    



    


