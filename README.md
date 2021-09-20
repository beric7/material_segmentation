# material_segmentation
structural material segmentation

Code for the semantic segmentation of structural material. This repository is in connection to the paper 
"Structural Datasets: A Contribution". 

The following code is included: 
- Pre-processing data
- Training
- Testing
- Prediction visualizations

Pre-processing data

We have included files to resize images, rescale segmentation masks, and randomly sort images into Testing and Training. Additionally, we have provided the code to run the labelme2016 processing step for converting JSON annotations and image pairs into segmentation masks and one-hot-encoded vector images. 

Training

In order to train the model follow the procedure outlined below:

During training there are model checkpoints at points defined during training. At these checkpoints one can test the current model on the validation data 

Testing
Once training has converged or when it has stopped, we can used the best checkpoint based on the validation data results. This checkpoint is loaded and our test data is evaluated. 

The code for capturing the performance metrics is here:

Prediction Visualizations:

The code for visualizing the predictions on images is here:

We have also provided code on how to concatenate the mask, overlays, and images to create prediction grids. 
