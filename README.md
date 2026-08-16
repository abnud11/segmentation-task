In this project, we use the [massachusetts buildings dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset), The dataset contains remote sensing image segmentation data split into training, validation and test.

We train two types of models: deep learning models from torchvision library and a transformer model using huggingface transformers library, the expirment is to show that the choice of the model affects the performance. Our hypothesis is that the transformer model will outperform even the ensemble learning of multiple torchvision models.

We ran the torchvision expirement by running:
```
python torchvision-expirement.py
```
The evaluation metrics we use is test loss and mIoU(mean Intersection over union). mIoU is a common metric used in segmentation tasks where the area of the intersection between the model segment and the labeled segment is divided by the area of the union, the closer the metric is to 1 the better the model and a value of 0.5 at the very least should be achieved for the model to be considered valid.

We also calcualte an IoU for each class in the massachusetts dataset, currently two classes: building and background.

The building class has smaller area than the background class, so it's expected that the IoU for the building will be smaller than the background IoU since it's more challenging for the model to learn the location of buildings than background.
The final results of the torchvision expirement is:
> Test loss=0.2463 | mIoU=0.7129 | background_IoU=0.8783 | building_IoU=0.5474

Now, let's run the transformers expirement:
```
python transformers-expirement.py
```
The final results of it is:
>Test metrics:
>100%|███████████████████████████| 3/3 [00:00<00:00,  5.83it/s]
>{'test_loss': 0.2510865330696106, 'test_mean_iou': 0.7043304239368607, 'test_background_iou': 0.8806780708510707, 'test_building_iou': 0.5279827770226507, 'test_runtime': 1.0119, 'test_samples_per_second': 9.882, 'test_steps_per_second': 2.965, 'epoch': 30.0}

To our surprise, the ensemble of two torchvision models outperformed the transformer model in all metrics except the background IoU(which is subjectively not important)

So our hypothesis that the transformer will outperform was wrong, in the future we may expirement with a different transformer model(we used the segformer model).

All code ran on this laptop specs:
AMD Ryzen 7 7850HS, Nvidia RTX 4050 GPU 6GB VRAM, 24GB RAM, NVMe SSD.

The code had to be adapted to fit in 6GB GPU RAM.