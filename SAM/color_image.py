r''' to obtain an image with in red the no usable lumen, in green the usable lumen and in black the nuclei
Autor Sonia '''
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import json

if __name__=='__main__':  
    # folder names definition
    path = os.getcwd()
    bilancio_folder='mask_analysis'
    bilancio_name='mask_tipe'
    folder_mask_name='SAM_out'
    folder_imput_name='patch'
    folder_mask='MASKS'

    for wsi in sorted(os.listdir(os.path.join(path,folder_mask))):
        print(wsi)

        with open(os.path.join(path,bilancio_folder,wsi,bilancio_name)) as file:
            data = json.load(file) # Load the contents of the JSON file as a Python object

        for patch_name in data.keys(): # for each patch of the wsi image analysed
  
            image = np.asarray(
                Image.open(
                    os.path.join(path,folder_imput_name,wsi,patch_name+'.jpg')
                ).convert("L")
                )       

            image_rgb = np.asarray(
                Image.open(
                    os.path.join(path,folder_imput_name,wsi, patch_name+'.jpg')
                ).convert("RGB")
                )
            output=image_rgb.copy()

            # load of SAM output
            masks = np.load(
                os.path.join(path, folder_mask_name,wsi,patch_name + ".npy"),
                allow_pickle=True,
            )

            sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)

            for i,item in enumerate(sorted_anns):
                mask_name='M_'+str(i)
                ann=np.array(item["segmentation"], dtype=int)

                if mask_name in data[patch_name]['isolated_lumes']:
                    output[ann==1]=[255,0,0] # in red the isolated lumen

                elif mask_name in data[patch_name]['good_lumes']: 
                    output[ann==1]=[0,255,0] # in green the usefull lumen

                elif mask_name in data[patch_name]['nuclei']: 
                    output[ann==1]=[0,0,0] # in black the usefull lumen

            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.title('original image')
            plt.subplot(1,2,2)
            plt.imshow(output)
            plt.axis('off')
            plt.title('colored image')
            plt.show()


                    
                        


