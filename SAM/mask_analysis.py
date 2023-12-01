r''' CODE TO PRODUCE A JSON FILE FOR EACH WSI IMAGE WITH THE 'CLASSIFICATION' OF THE MASK OF EACH EXTRACTED
Autor Sonia'''

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import json
          

# folder names
path = os.getcwd()
mask_kind='MASKS'
folder_mask_name = "SAM_out"
folder_imput_name='patch'
bilancio_folder='mask_analysis'

up_threshold=229638 # 21,9% of the size of the image
down_threshold=1258 # 0.12% of the size of the image
nuclei_threshold=137

if not os.path.isdir(os.path.join(path,bilancio_folder)):
        os.mkdir((os.path.join(path,bilancio_folder)))
        
bilancio_name='mask_tipe'
tree_folder='tree'
total_lume={}
    
for wsi in sorted(os.listdir(os.path.join(path,tree_folder))): # i use only the image about which i have the tree structure
    print(wsi)

    if wsi.split('-')[3][0:3]=='PAS': # lume_treshold change in order to the staining 
        lume_threshold=0.75
    else:
        lume_threshold=0.63    


    if not os.path.isdir(os.path.join(path,bilancio_folder,wsi)):
        os.mkdir((os.path.join(path,bilancio_folder,wsi)))
    if not os.listdir(os.path.join(path,bilancio_folder,wsi)):
        bilancio={}
    else:    
        with open(os.path.join(path,bilancio_folder,wsi,bilancio_name)) as file:  # Load the contents of the JSON file as a Python object
                bilancio = json.load(file) 

    for patch_name in sorted(os.listdir(os.path.join(path,mask_kind,wsi))):
        
        if patch_name in bilancio.keys():
            continue
            # definizione delle variabili specifiche per ogni immagine
      
        bilancio[patch_name]={}
        good=0
        mask_good=[]
        bad=0
        mask_bad=[]
        usable=0
        mask_use=[]
        no_usable=0
        mask_no_use=[]
        mask_big=[]
        n_nuclei=0
        nuclei=[]

        # load of the original image in grey scale not using averaging
        image = np.asarray(
        Image.open(
            os.path.join(path,folder_imput_name,wsi, patch_name+'.jpg')
        ).convert("L")
        )


        massimo=np.max(image) #maximum image value in gray scale (to be used for locating lumens)
            
        # load of the mask created by SAM
        masks = np.load(
            os.path.join(path, folder_mask_name,wsi,patch_name + ".npy"),
            allow_pickle=True,
        )
        sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        lim = len(sorted_anns)



  
        # load of the edge and the values of the tree structure
        print(f'image: {patch_name}')


        with open(os.path.join(path,tree_folder,wsi,patch_name,'nodi'), "r") as f:
            edge=json.load(f) 

        with open(os.path.join(path,tree_folder,wsi,patch_name,'values'), "r") as f:
            livel=json.load(f)

   



        for i,item in enumerate(sorted_anns):
            ann=np.array(item["segmentation"], dtype=int) 
            if np.sum(ann)>down_threshold:# limit to avoid considering masks too small
                if np.sum(ann)<up_threshold: # limit to avoid considering back ground masks

                #-----------------------lumes analisy--------------------------------------------------

                    if (np.mean(image[np.where(ann==1)])/massimo)>lume_threshold: 
                            # print(f'M_{i}\n')
                            conf=[]
                            #--------------find the mask to which the lume is connected--------------------
                            for j in edge:
                                if j[1]==f'M_{i}':
                                    if j[0]==patch_name: # lume connected to the root mask
                                        bad+=1
                                        mask_bad.append(f'M_{i}')
                                        # print('-connected to the root')
                                        break
                                    else:
                                        flag=0
                                        conf.append(int(j[0].split('_')[1]))
                                        while flag==0:# the while loop allows me to save in conf all the masks containing the analysed lumen. it stops at the moment when the root is reached
                                            for k in edge:
                                                if k[1]==f'M_{conf[-1]}': 
                                                    if k[0]!=patch_name:
                                                        conf.append(int(k[0].split('_')[1]))
                                                    else :
                                                        flag=1    

                                        # find the largest mask to which the lumen is attached (excluding back ground masks masks)
                                        flag2=0
                                        if len(conf)>1:
                                            while flag==1:
                                                if conf==[]: # if it is empty, it means that the piece of code below has removed every element, so it means that 
                                                    # the lumen I am analysing is only connected to macro masks
                                                    bad+=1
                                                    mask_bad.append(f'M_{i}')
                                                    # print(f'-connected to a back ground mask\n')
                                                    flag=0
                                                    flag2=1
                                                else:    
                                                    n_mask=np.min(conf) # value of the largest mask to which the lumen is connected
                                                    if np.sum(np.array(sorted_anns[int(n_mask)]["segmentation"], dtype=int))>up_threshold: 
                                                        conf.remove(n_mask)
                                                    else:
                                                        # print(f'-connected  to M_{int(n_mask)}\n')
                                                        flag=0
                                        else:
                                            n_mask=np.min(conf) # if conf has only one element I don't have to do any checking and I already know the mask it is linked to  
                                            # print(f'-connected to M_{int(n_mask)}{n')       
                                           
                                            if np.sum(np.array(sorted_anns[int(n_mask)]["segmentation"], dtype=int))>up_threshold: # l'unica maschera a cui é collagata é una macro maschera
                                                bad+=1
                                                mask_bad.append(f'M_{i}')
                                                flag2=1
                                                # print(f'-connected to a back ground mask \n')
                                    
                                        if flag2==0: # if flag2=0 it means that the lumen I am analysing is connected to a mask other than a macro
                                            if  ((np.mean(image[np.where(np.array(sorted_anns[int(n_mask)]["segmentation"], dtype=int)==1)]))/massimo)>lume_threshold: # the larger mask to which it is attached is a lumen
                                                bad+=1
                                                mask_bad.append(f'M_{i}')
                                                # print(f'-connected to a lumen\n')
                                            else:    
                                                good+=1
                                                mask_good.append(f'M_{i}')
                                                mask_big.append([f'M_{i}',f'M_{n_mask}'])
                                                # print(f'-good\n')





            #-----------------------------I find masks that have at least one element in them and that are not lumen---------------------
                    else:
                        edge_in=[]

                        
                        for z in edge:
                            if z[0]==f'M_{i}':
                                edge_in.append(z[1])
                            elif z[1]==f'M_{i}': # analysing the mask to which it is connected
                                mask_up=z[0]
                        if len(edge_in)!=0:
                            usable+=1
                            mask_use.append(f'M_{i}')
                        else:
                                if mask_up==patch_name:
                                    #print(f'mask M_{i} has no elements inside it and is connected to the root')
                                    no_usable+=1
                                    mask_no_use.append(f'M_{i}')
                                else:
                                    if np.sum(np.array(sorted_anns[int(mask_up.split('_')[1])]["segmentation"], dtype=int))>up_threshold:
                                       #print(f'mask M_{i} has no elements inside it and is linked to a back ground mask')
                                        no_usable+=1
                                        mask_no_use.append(f'M_{i}')

            else:
                if (np.mean(image[np.where(ann==1)]))<nuclei_threshold: # threshold for the nuclei identification
                    n_nuclei+=1
                    nuclei.append(f'M_{i}')
          
                        
# save of the mask classification in a json file

        bilancio[patch_name]['N_isolated_lumes']=bad # number of isolated lumen
        bilancio[patch_name]['N_good_lumes']=good # number of usable lumen
        bilancio[patch_name]['N_full_mask']=usable # number of mask with more mask inside
        bilancio[patch_name]['N_empty_mask']=no_usable # number of mask with nothing inside
        bilancio[patch_name]['N_nuclei']=n_nuclei # number of nuclei
        bilancio[patch_name]['isolated_lumes']=mask_bad # list of isolated lumen
        bilancio[patch_name]['good_lumes']=mask_good # list of usable lumen
        bilancio[patch_name]['full_mask']=mask_use # list of mask with more masks inside
        bilancio[patch_name]['empty_mask']=mask_no_use # list of masks with nothing inside
        bilancio[patch_name]['big_mask']=mask_big # list [usable lumes,big mask in wich is insered]
        bilancio[patch_name]['nuclei']=nuclei # list of nuclei masks
        with open(os.path.join(path,bilancio_folder,wsi,bilancio_name), "w") as f:
            json.dump(bilancio, f,indent=4)

        print(f"{int(len(bilancio.keys()))}/{int(len(os.listdir(os.path.join(path, mask_kind,wsi))))}")

  