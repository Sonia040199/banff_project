r''' divides the wsi image in patched, saves them in .jpg format and selects the usable ones.
Autor Sonia'''
import os
import numpy as np
import openslide
import matplotlib.pyplot as plt


# function to discart images
def filter(img, dim):  # filter out background patches
    std = np.array(img)[:, :, 0].std(), np.array(img)[:, :, 1].std(), np.array(img)[:, :, 2].std()
    m = max(abs(std[0] - std[1]), abs(std[2] - std[1]), abs(std[0] - std[2])) # used to discart monocromatic images
    k = np.sum(np.array(img).mean(axis=2) > 220) / (dim * dim) # used to know the percentage of with back ground in the image
    z=np.array(img).min(axis=2).mean() # used to eliminate to shine images ( so the artefacts)

    if z > 240 or m < 4.5 or k > 0.6: 
        return True
    return False





# READ IMAGE

path = os.getcwd()
folder_name = "big_data" # forder with the wsi images
patch_folder='patch' # forder for the extracted patches
counter = 0
dim = 1024 # dimension of the patches

if not os.path.isdir(os.path.join(path, patch_folder)): # patch folder creation
    os.mkdir(os.path.join(path, patch_folder))


for k in sorted(os.listdir(os.path.join(path,folder_name))):
    file=k.split('.')[0]

    print(f'PROCESSING: {file}')

    if not os.path.isdir(os.path.join(path, patch_folder,file)): # wsi forlde in patch folder creation
        os.mkdir(os.path.join(path,patch_folder,file))


    wsi = openslide.open_slide(os.path.join(path, folder_name, file + ".svs")) # reading wsi image

    # patches division and filtering
    i = 0
    while i < wsi.level_dimensions[0][0]:
        j = 0
        while j < wsi.level_dimensions[0][1]:
            img = wsi.read_region((i, j), 0, (dim, dim))

            if filter(img, dim):
                j += dim
                continue
            img = img.convert('RGB')
            img.save(os.path.join(path, patch_folder , file, f"image_x{i}_y{j}" + ".jpg")) # saving of the image and the it's position in the wsi image
            

 
            j += dim
        i += dim

        if i > wsi.level_dimensions[0][0] - dim:
            if i < wsi.level_dimensions[0][0] - 50:
                i = wsi.level_dimensions[0][0] - dim
            else:
                i += dim



    a=len(os.listdir(os.path.join(path, folder_name)))
    counter+=1             
    print(f'{counter}/{int(a)} IMAGED DONE')
