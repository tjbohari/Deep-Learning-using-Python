# Object Classification and Localization

The objective of this project is binary classification of images and localizing the object in an image. We try to find highly correlated neurons through technique 
of average pooling layer. After getting the highly corrrelated neurons, we try to find if we can find the neurons which causes the convulational layer to classify them 
into particular category.


## Object Classification

### Dataset preparation and preprocessing: 
- The images of bird and cats for the dataset are scraped from [Bulk-Bing-Image-downloader](https://github.com/ostrolucky/Bulk-Bing-Image-downloader).
- 500 images of both the classes are used for training the model. Additionally, data augmentation is used to create artificial data and avoid overfitting of the model.
- Preprocessing is done using ImageDataGenerator from tensorflow. 

### Model Training
- Model is trained using the dataset prepared.
- An accuracy of 99% is achieved with only 9 epochs.

## Object Localization

To perform object localization, technique of [Class Activation Mapping](http://cnnlocalization.csail.mit.edu/) is used to activate the neurons with high weights. 
The CAM model highlights the neurons with higher weights in the image which aids in localizing the object in the image.
In our case, we have calculated correlation between neurons and adjusted the weights accordingly for CAM. We were able to successfully localize the objects in the images.

# Contributors
- [Mahewash Abdi](https://github.com/mahewashabdi)
- [Taha Husain](https://github.com/tjbohari/)
