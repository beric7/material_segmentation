# Material Segmentation

<p align="center">
    <img src="/figures/ex 3.jpeg" | width=800/>
    <img src="/figures/ex 1.jpeg" | width=800/>
</p>

The four semantic classes in the dataset are:
```
Background
Structural Concrete
Structural Steel 
Metal Decking
```
***Coming soon in November***
:red_circle:\[[Paper](/access/not_ready.png)\] :red_circle:\[[Dataset](/access/not_ready.png)\] :red_circle:\[[Trained models](/access/not_ready.png)\]

The structural material segmentation dataset can be used for auxiliary structural inspection tasks to aid in the localization of structural damage, provide context to predictions, and for more futuristic style transfer [SPADE](https://arxiv.org/abs/1903.07291) and [GAN](https://arxiv.org/abs/1912.04958) / [GAN-Inversion](https://arxiv.org/abs/2101.05278) applications. 

## Results

<p align="center">
    <img src="/figures/result_table.jpg" | width=600/>
    <img src="/figures/confusion matrix.png" | width=400/>
</p>

## Requirements
The most important environment configurations are the following:
- Pytorch >= 1.4
- Python >= 3.6
- tqdm
- matplotlib
- sklearn

## Evaluating the Trained DeeplabV3+ Model
- Download the DeeplabV3+ :red_circle:[trained model weights](/access/not_ready.png)
- Configure ***run_metrics_evaluation.py***

You will get the f1 score, the jaccard index, and the confusion matrix. We suggest running this in an IDE. 
  
## Visualizing the results from the Trained DeeplabV3+ Model
Once training has converged or when it has stopped, we can used the best checkpoint based on the validation data results. This checkpoint is loaded and our test data is evaluated. 

***run_show_results__.py***
- gets predicted masks
- gets combined mask and image overaly
- gets one-hot-encoded vector images of predictions

## Training with the Structural Material dataset

1. Clone the repository
2. Download the :red_circle:[dataset](/access/not_ready.png)
3. Go into the Training folder
4. Create a DATA folder
5. Copy and paste the Train and Test folders for 512x512 images from the dataset you downloaded into the DATA folder
6. The DATA folder should have a folder called 'Train' and a folder called 'Test'. Inside each of those folders include the mask and image pairs in their respective folders (Masks, Images). 
7. If you have set this up correctly then you are now ready to begin.

Neccesary and optional inputs to the ***main_plus.py*** file:
('-' means it is neccessary, '--' means that these are optional inputs)
```
 -data_directory = dataset directory path (expects there to be a 'Test' and a 'Train' folder, with folders 'Masks' and 'Images')
 -exp_directory = where the stored metrics and checkpoint weights will be stored
 --epochs = number of epochs
 --batchsize = batch size
 --output_stride = deeplab hyperparameter for output stride
 --channels = number of classes (we have four, the default has been set to four). 
 --class_weights = weights for the cross entropy loss function
 --folder_structure = 'sep' or 'single' (sep = separate (Test, Train), single = only looks at one folder (Train). If you want to get validation results instead of getting back your test dataset results then you should use 'single'. If you want to test directly on the Test dataset then you should use 'sep'.
 --pretrained = if there is a pretrained model to start with then include the path to the model weights here. 
```

Run the following command:
(there is an example shell script file, 'var_8plus_40epoch_3-2021_ca.sh' we used for training provided in the 'Training' folder).
```
python main_plus.py -data_directory '/PATH TO DATA DIRECTORY/' -exp_directory '/PATH TO SAVE CHECKPOINTS/' \
--epochs 40 --batch 2
```

During training there are model checkpoints saved every epoch. At these checkpoints the model is compared against the test or validation data. If the test or validation scores are better than the best score, then it is saved. 

## Training with a custom dataset
1. Clone the repository
2. Ensure your image and mask data is 512x512 pixels. *(can use the ***rescale_image.py*** in Pre-processing)*
3. Ensure that if you resized your masks to 512x512 that they did not interpolate the colors into more color classes than you have. The expected format is BGR. *(can use the ***rescale_segmentation.py*** in Pre-processing)*
4. You now need to go into the ***datahandler_plus.py*** file and edit the colors as necessary. For example, the Structural Materials dataset used the following format, which is in the ***datahandler_plus.py*** in this repository.
```
# color mapping corresponding to classes
# ---------------------------------------------------------------------
# 0 = background (Black)
# 1 = Steel (Red)
# 2 = Concrete (Green)
# 3 = Metal Deck (Yellow)
# ---------------------------------------------------------------------
self.mapping = {(0,0,0): 0, (0,0,128): 1, (0,128,0): 2, (0,128,128): 3}
```
6. Adjust the number of 'channels' in the training command to match the number of channels that you have.
7. Ensure that your DATA folder has a folder called 'Train' and a folder called 'Test'. Inside each of those folders include the mask and image pairs in their respective folders (Masks, Images). 
8. If you have set this up correctly then you are now ready to begin.

## Building a Custom Dataset
(The images in the dataset were annotated using [labelme](https://github.com/wkentaro/labelme). We suggest that you use this tool)

1. Before beginning to annotate, we suggest that you use jpeg for the RGB image files. We advised against beginning with images which are already resized. 

2. After annotating you will have matching JSON and jpeg files, indicating the annotation and image pair respectfully. 

3. You will take these files and generate masks and one-hot-encoded vector files using ***run_labelme2voc_.py*** file in Pre-processing. Then you can re-scale these images and masks using the respective files in Pre-processing. You can also use the random sort function we have created to randomly split the data. 

The ***labels_corrosion_segmentation.txt*** file contains the class labels needed for the ***run_labelme2voc_.py*** function. If your classes are different then they need to be reflected in this particular file.

## Citation
```
Structural Material Dataset: 
Structural Material Model:
Paper:
```


