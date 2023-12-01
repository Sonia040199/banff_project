TO IMPLEMENT THE BEST MODEL WITHOUT THE GT IS NECESSARY TO:
1. ADD THE IMAGES TO CLASSIFY IN BANFF\lumen_classification\DATA_SET\new_images\patches
2. ADD THE MASK OF THE NUCLEI IN BANFF\lumen_classification\nuclei_mask
3. EXECUTE THE CODE new_image_feature_extraction.py in BANFF\lumen_classification
4. EXECUTE THE CODE new_image_implementation.py in BANFF\lumen_classification

TO TRAIN CHANGING THE FEATURE:
1. ADD THE IMAGES TO CLASSIFY IN BANFF\lumen_classification\DATA_SET\new_images\patche
2. ADD THE MASK OF THE NUCLEI IN BANFF\lumen_classification\nuclei_mask
3. EXECUTE THE CODE new_image_feature_extraction.py in BANFF\lumen_classification
4. EXECUTE THE CODE lumen_classification.py in BANFF\lumen_classification changing the features
5. EXECUTE THE CODE lumen_class_model_application.py in BANFF\lumen_classification

TO IMPLEMENT THE BEST MODEL WITH THE GT IS NECESSARY TO:
1. ADD THE IMAGES TO CLASSIFY IN BANFF\lumen_classification\DATA_SET\new_images\patche
2. ADD THE MASK OF THE NUCLEI IN BANFF\lumen_classification\nuclei_mask
3. EXECUTE THE CODE lumen_labeling.py BANFF\lumen_classification and upgraid the excel file
4. EXECUTE THE CODE lumen_feature_extraction IN BANFF\lumen_classification changing the value of the variable set_folder with set_folder_new
5. EXECUTE THE CODE lumen_class_model_application IN BANFF\lumen_classification changing the value of the variable set_folder with set_folder_new

------------------------------------------------WHAT THERE IS IN THE FOLDERS-------------------------------------------------------------------
-GENERAL FOLDER:
	- CODES
	-- image_selection.py:  divides the wsi image in patched, saves them in .jpg format and selects the usable ones.
	IS THE FIRST CODE TO EXECUTE

	- FOLDERS
	-- big_data: folder with the WSI images

nuclei_classification FOLDER:
	-CODES
	--lumen_mv_class: to classify the lumen using the majority voting applied to the classification of the nuclei
	--nuclei_classification.py: for the classification of the nuclei. it save the trained model in the folder models
	--nuclei_feature_exctraction: for the feature extraction of the nuclei. It saved the matrix of the feature extraction, the nuclei included in a crown and the grpund thruth in the set folder
	order to use the codes:
	1.nuclei_feature_exctraction.py
	2.nuclei_classification.py
	3.lumen_mv_class.py

	- FOLDERS
	--DATA_SET: folder with a folder for the train set and a folder for the test set. each fodler contain a folder with the patches and the data from the feature extraction
	--models: folder with the trained models
	--nuclei_mask: mask of the nuclei

lumen_classification FOLDER:
	- CODES
	-- excel_extractor.py: to check the labels written in the excel file
	-- lumen_labeling.py: to complete the excel file with the GT
	-- feature_rappresentation.py: to implement the graphycs feature selection
	-- lumen_feature_extraction.py: to extract the feature from the data set. The code will save in the same folder of the data set the feature matrix, the gt, the name of the extracted feature and a list of list with the usable lumen.
	-- lumen_class_model_application.py: code to apply the trained models to PAS, TRI images, fake PAS and worst case ones.
	-- lumen_classification.py: code for the train, the validation and the test of the models
	-- distance_SVM.py: to obtain the distance between the element to classify and the plan use by SVM model to classify them
	-- manual_FS.py: code to analyse the performances of the models considering one feature per time. The code save the performances in an Excelle file and save the selected feature for each classifier. 
	-- new_image_feature_extraction.py: to implement the feature extraction on new images without the ground thruth
	-- new_image_implementation.py: to implement the classification on new images without the ground thruth
	order to use the codes:
	1.lumen_labeling.py
	2.excel_extractor.py
	3.lumen_feature_extraction.py
	4.manual_FS.py (if you want to implement an automatic feature extraction)
	5.feature_rappresentation.py (if you want to implement a graphyc feature extraction)
	6.lumen_classification.py
	7.lumen_class_model_application.py
	8.distance_SVM.py

	-FOLDERS
	-- DATA_SET: contain the chosen patches for the train, the test and the validation set. there is also another folder with the worst case patches, the tri stained images and the fake pas images
		in each sub folder there is the feature matrix, the GT of the element and the usable lumen
	-- nuclei_mask: mask of the nuclei
	-- trained_model: trained model, the name of each file specify the name of the ML model and the number of selected feature
	-- feature_selected: the manual selected feature for each classifier. 
	-- label: contain the labeled patches. The labeling was done by the pathologist

	- OTHERS
	-- lumen_labels.xlsx: excel file with the GT.
	-- feature_selection.xlsx: ec=excel file for the rappresentation of the performances of each feature applied once by once. 


SAM FOLDER:
	- CODES
	-- mine_sam.py: implements the SAM segmentation and to creates the tree structure with the obtained masks
	-- mask_analysis.py: analysis of the mask and first classification to find lumes ( usable and not) and nuclei
	-- color_image.py: to obtain an image with in red the no usable lumen, in green the usable lumen and in black the nuclei
	order to use the codes:
	1.mine_sam.py
	2.mask_analysis.py
	3.color_image.py
	
	- FOLDERS
	-- patch: folder with selected patches
	-- MASKS: folder with SAM output masks
	-- SAM_out: folder with SAM segmentations
	-- tree: folder with the edges and the values for the tree structure
	-- mask_analysis: folder with one json file each WSI image. In each json file there is the analysis of the mask for each patch of the wsi image

	- OTHERS
	-- sam_vit_h_4b8939.pth: SAM model

requirements-test --> for the codes in SAM folder and for image_selection.py
requirements--> for the others codes