# Material Segmentation

<p align="center">
    <img src="/figures/ex 3.jpeg" />
    <img src="/figures/ex 1.jpeg" />
</p>

The four semantic classes in the dataset are:
```
Background
Structural Concrete
Structural Steel 
Metal Decking
```

\[[Paper]()\] \[[Dataset]()\] \[[Trained models]()\]

The structural material segmentation dataset can be used for auxiliary structural inspection tasks to aid in the localization of structural damage, provide context to predictions, and for more futuristic style transfer [SPADE](https://arxiv.org/abs/1903.07291) and [GAN](https://arxiv.org/abs/1912.04958) / [GAN-Inversion](https://arxiv.org/abs/2101.05278) applications. 

## Requirements
The most important environment configurations are the following:
- Pytorch == 1.4
- Python == 3.6
- 
## Evaluating Trained DeeplabV3+ Model
- Download the DeeplabV3+ [trained model weights](10.7294/16624495)
  
## Testing the Trained DeeplabV3+ Model
Once training has converged or when it has stopped, we can used the best checkpoint based on the validation data results. This checkpoint is loaded and our test data is evaluated. 

The code for capturing the performance metrics is here:

Prediction Visualizations:

The code for visualizing the predictions on images is here:

We have also provided code on how to concatenate the mask, overlays, and images to create prediction grids. 

## Training with the Structural Material dataset

- Clone the repository
- Download the [dataset]()

During training there are model checkpoints at points defined during training. At these checkpoints one can test the current model on the validation data 

## Training with a custom dataset
1. Clone the repository


## Citation
```
hahasadf
```


