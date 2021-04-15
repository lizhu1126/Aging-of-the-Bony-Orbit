# Aging-of-the-Bony-Orbit
# 1.BackGround
This code is used to realize the automatic segmentation of the orbital region of the facial and skull image, and then perform automatic aging feature representation and classification of the segmented orbital contour. The craniofacial image dataset should be prepared of skull CT scanning data in DICOM format. You can use this code to quickly verify larger samples.

# 2.Requirements
- Ubuntu18.04
- python 3.8
- pytorch 1.7
- cuda 11.0
- anaconda 4(recommended to use
- OpenCV (4.0+)

# 3.Usage
## 3-1.verification
first

```cd verification/```

In folder "input", there is a craniofacial image of a young male named "test.png". You can use it to verify the models I trained in the folder "models". Use the following code:

```sh run.sh```

It will print "young" or "mid-age" or "old-age". The result of this image we expect is "young". You can train your own models to replace the ones I trained. 

## 3-2.train
Make your own dataset. Make sure the image size is 600*360.
### 3-2-1 U-Net
first

```cd train/segmentation/```

Your dataset should be placed in the folder of "DATA", including origin images and masks of one channel. Then you can train your own U-Net.

```python train.py```

after training, the model file will be saved in the folder of "checkpoints".

### 3-2-2 AlexNet
Crop the image size in your dataset to 360*360.
first

```cd train/classification/```

Your dataset should be placed in the folder of "DATA". Each category should be distinguished. Then you can train your own classification network.

```python main.py```

After training, the model file will be saved as "AlexNet.pth".

## 3-3.calcuFeature
first

```cd train/calcuFeature/```

Here you can use the following code to calculate the area, height, circumference and roundness of an orbital image."test.png" is an orbital image after segmentation.

```python connectedComponent.py```

# 4.Refered Repos
Our work is not built from scratch. Great appreciation to these open works！

[U-Net](https://github.com/milesial/Pytorch-UNet)

[AlexNet](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test2_alexnet)









