r''' implements the SAM segmentation and to creates the tree structure with the obtained masks
Autor Sonia'''
import torch
import cv2
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
import matplotlib.pyplot as plt
import slideio
import numpy as np
import tifffile as tiff
import json


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("used devices: ", DEVICE)


def show_anns(anns): 
    ''' function to create an image with all the mask produced by SAM
input: SAM output
output: image with all the mask producted by SAM'''


    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True) # sorted of the mask
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann["segmentation"] # extraction of the mask
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0] # color the mask in the RGB image with a random color
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))

def level_mask(path,folder_out_name,wsi,input_name,image,sorted,lim,i): 
    '''to save each mask in a folder with the name of the patch
input:
- path: path of the main foler in wich there is the code
- folder_out_name: main folder where the mask are saved
- wsi: name of the wsi image
- input_name: patch name
- image: RGB image of the patch
- sorted: sorted SAM output
- lim: number of mask that to save
- i: position of the mask in sorted
output:
- saving of the mask in the created folder '''

    if not os.path.isdir(os.path.join(path,folder_out_name,wsi,input_name)): # creation of the folder with the name of the patch 
        os.mkdir((os.path.join(path,folder_out_name,wsi,input_name)))

    ann=sorted[i]['segmentation']    

    mask = np.zeros_like(image)
    mask[ann] = (255, 255, 255)
    single_mask = np.where(mask != 0, 0, image)
    Image.fromarray(single_mask).save(
        os.path.join(path,folder_out_name,wsi,input_name, "M_" + str(lim-i-1) + ".jpg"),
    )

def tree_creator(path,folder_out_name,tree_folder,image,masks,patch_name,wsi):
    '''function to create the three structure 
input:
- path: path of the main foler in wich there is the code
- folder_out_name: main folder where the mask are saved
- tree_folder: folder where the node and the values of the node of the tree are saved
- image: RGB patch image
- masks: SAM output
- patch_name: name of the patch
- wsi: name of the wsi image '''

    if len(masks) == 0:
        return

    root = patch_name # the name of the patch is the root of the tree

    # sorting of SAM output in crescent order considering the size of the mask
    sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True) 
    sorted_prova = sorted(sorted_anns, key=(lambda x: x["area"]))

    lim=len(sorted_prova) # number of mask involve in the tree structure if lim=len(sorted_prova) you are considering all the mask
    # initialisation of edge and values
    edge_list = [] 
    values = {} 
    values[root] = 0 # the root has value=0


    for i, item_i in enumerate(sorted_prova):
        ann_i = np.array(item_i["segmentation"], dtype=int) # extraction of each mask
        common_pixel_vector = [] 
        for j, item in enumerate(sorted_prova):
            if j <= i: # to avoid to compare the mask with the smaller ones
                continue
            ann_j = np.array(item["segmentation"], dtype=int) # mask to compare
            common_pixel = ann_i * ann_j # evaluation of the common pixel
            common_pixel_vector.append(np.sum(common_pixel)) 
            # each element of common_pixel_vector contains within it the pixels that the i-th mask has in common with all the other masks used for comparison
            # e.g. the fourth element of wild contains within it the sum of pixels common between the i-th mask and the i+4 mask
        common_pixel_vector = np.array(common_pixel_vector) 

        # if the i-th mask has no pixels in common with any mask or if common_pixel_vector is empty 
        # (the latter condition only occurs if the i-th mask is the last) then the mask I am analysing is directly connected to the root node
        if np.all(common_pixel_vector == 0) or common_pixel_vector == []: 
            edge_list.append((root, f"M_{lim-1-i}"))
            level_mask(path,folder_out_name,wsi,patch_name,image,sorted_prova,lim,i) # saving of the i-th mask
        else:
            reverse_common_pixel_vector = common_pixel_vector[::-1]

            reverse_common_pixel_vector[reverse_common_pixel_vector == 0] = ann_i.shape[0] * ann_i.shape[0]  # to set all elements equal to zero equal to a very high value

            ind = np.where(reverse_common_pixel_vector == np.min(reverse_common_pixel_vector))[0] # find the position of the j-th mask that has the least number of pizels in common with the i-th mask
            if len(ind) == 1:
                ind_min = int(ind) 
            else:
                ind_min = int(ind[-1]) # if there is more than one element equal to the minimum I take the last one
            edge_list.append((f"M_{ind_min}", f"M_{lim-1-i}")) # I add to the edge_list the pair of the two masks that are connected, putting the larger mask first
            level_mask(path,folder_out_name,wsi,patch_name,image,sorted_prova,lim,i) # saving of the i-th mask

    # to apply lavels
    comparison = [root]
    while len(values) <= len(edge_list): 
        for i in edge_list:
            for j in comparison:
                if i[0] == j: 
                    values[i[1]] = values[j] + 1 # the layer of the attached mask is the same as the upper mask plus one
                    comparison.append(i[1])
                    break            
    # saving of the edge and levels
    if not os.path.isdir(os.path.join(path,tree_folder,wsi,patch_name)):
        os.mkdir((os.path.join(path,tree_folder,wsi,patch_name)))
    with open(os.path.join(path,tree_folder,wsi,patch_name,'nodi'), "w") as f:
        json.dump(edge_list, f,indent=4) 

    with open(os.path.join(path,tree_folder,wsi,patch_name,'values'), "w") as f:
        json.dump(values, f,indent=4)

    #------------ decoment to visualise the tree structure---------------
    # G = nx.Graph()
    # G.add_edges_from(edge_list)
    # nx.set_node_attributes(G, values=values, name="subset")
    # pos = nx.multipartite_layout(G, subset_key="subset", scale=4)
    # plt.figure(1)
    # nx.draw_networkx(G, pos=pos, with_labels=True, node_color="lightblue")
    # plt.axis("off")
    # plt.show()
    #--------------------------------------------------------------------


if __name__ == "__main__":
    path = os.getcwd()
    folder_name = "patch"
    folder_out_name = "MASKS" # folder to save each singular mask
    folder_save = "SAM_out" # folder to save the sam output
    tree_folder='tree' # folder to save the three structure

    # folder creation
    if not os.path.isdir(os.path.join(path, folder_out_name)):
         os.mkdir(os.path.join(path, folder_out_name))

    if not os.path.isdir(os.path.join(path, folder_save)):
        os.mkdir(os.path.join(path, folder_save))

    if not os.path.isdir(os.path.join(path,tree_folder)): 
        os.mkdir((os.path.join(path,tree_folder)))

    for wsi_image in os.listdir(os.path.join(path, folder_name)):
        print(f'processing wsi: {wsi_image}')

        if not os.path.isdir(os.path.join(path, folder_out_name,wsi_image)):
            os.mkdir(os.path.join(path, folder_out_name,wsi_image))

        if not os.path.isdir(os.path.join(path, folder_save,wsi_image)):
                os.mkdir(os.path.join(path, folder_save,wsi_image))
        
        if not os.path.isdir(os.path.join(path,tree_folder,wsi_image)):
                os.mkdir((os.path.join(path,tree_folder,wsi_image)))


        for input_name in sorted(os.listdir(os.path.join(path, folder_name,wsi_image))):
            print(f'patch:{input_name}' )

            #image load
            image = np.asarray(
            Image.open(
                os.path.join(path, folder_name,wsi_image, input_name)
            ).convert("RGB")
        )

        # SEGMENTATION
            MODEL_TYPE = "vit_h"
            CHECKPOINT_PATH = r"C:/Users/sonia/segment-anything/sam_vit_h_4b8939.pth"
            sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
            sam.to(device=DEVICE)
            print("Now segmenting. Please wait.")
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=90,
                points_per_batch=64,
                pred_iou_thresh=0.9,
                #stability_score_thresh=0.96,
                stability_score_offset=1.2,
                box_nms_thresh=0.8,
                crop_n_layers=2,
                crop_nms_thresh=0.8,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=60,
                )
            masks = mask_generator.generate(image)
            print("segmentation done")
           
            np.save(os.path.join(path, folder_save,wsi_image, input_name.split('.')[0]), masks) # saving of the SAM output

            # single mask
            print('tree creation')
            tree_creator(path,folder_out_name,tree_folder,image,masks,input_name.split(".")[0],wsi_image)
            print('done')

            show_anns(masks)


            print(f"PATCH {int(len(os.listdir(os.path.join(path, folder_out_name,wsi_image))))}/{int(len(os.listdir(os.path.join(path, folder_name,wsi_image))))}  PROCESSED")

