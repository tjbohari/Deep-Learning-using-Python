
# Object Detection and Classification using YOLO and Mobilenet

#### -- Project Status: [Completed]

## Project Intro/Objective

In this project I have performed object detection and object classification. For the purpose of object detection I have used Tiny YOLO v4 and openCV library. YOLO works on the concept of you only look once, where it captures all the detected images and generates confidence for each detection. Using the confidence and NMS suppression, high quality objects are detected. Later, object classification is performed on the images detected from the object detection model. The model is a pretrained model from keras, *Mobilenet* where it is trained further on the dataset. The model classifies the cars into Sedan or SUV types. This model can be further extended to perform deeper classification.

### Technologies
* Python
* Keras
* Jupyter
* OpenCV
* Darknet

### Dependencies
- cv2
- numpy
- PIL
- os
- matplotlib
- tensorflow
- keras
- pandas

The project is divided in two queries where Q1 performs the object detection and Q2 performs the classification task. Below is the pipleline design for the project.

![Pipleline Design](https://github.com/tjbohari/Deep-Learning-using-Python/blob/main/Object%20Detection%20and%20Classification/Results/pipeline_design.png)


## Object Detection

To perform object detection, YOLO model is used with openCV library. The model takes input a frame on which object detection needs to be performed and returns the bounding boxes and the confidence score for each object detected. Furthermore, using the bounding box coordinates the images of detected objects are cropped and supplied to the object classification model. 

## Object Classsification

Object classification is performed using the Mobilenet model provided by Keras. For classification, I have used method of transfer learning for faster training and best performance. The model is pretrained on coco dataset. I have used the initial layers and trained it further on my dataset so it classifies them with high accuracy. Following are the steps included in training the model

### Dataset preparation and preprocessing
- The dataset is scraped from the web using bing image downloader.
- Some manual cleaning was required as unwanted images were also downloaded.
- 500 images from each class were used in training the model
- The images were resized using OpenCV library and normalized for the training of model.

### Model training
- Mobilenet V2 model was loaded from the tfhub library.
- Dropout layer to avoid overfitting.
- Softmax layer with size 2 to classify the images.

## Results

Below are the results from Q1 and Q2 for different frames. In Q1, boxes are drawn around the objects detected. In Q2, detected objects are classified into Sedan or SUV.

##### Q1

![Frame 1](https://github.com/tjbohari/Deep-Learning-using-Python/blob/main/Object%20Detection%20and%20Classification/Results/frame1.png)
![Frame 2](https://github.com/tjbohari/Deep-Learning-using-Python/blob/main/Object%20Detection%20and%20Classification/Results/frame2.png)

##### Q2
![Frame 1](https://github.com/tjbohari/Deep-Learning-using-Python/blob/main/Object%20Detection%20and%20Classification/Results/frame_Q21.png)
![Frame 2](https://github.com/tjbohari/Deep-Learning-using-Python/blob/main/Object%20Detection%20and%20Classification/Results/frame_Q22.png)

## Scores and Statistics

- F1 score for Q1: F1 score for Q1 (object detection) is 0.88.
- F1 score for Q2: F1 score for Q2 (object classification) is 0.70.
- Throughput for Q1: 6 frames per second.
- Throughput for Q2: 2.5 frames per second.

## Performance

FPS throguhput for Q1 and Q2 combined.

![throughput](https://github.com/tjbohari/Deep-Learning-using-Python/blob/main/Object%20Detection%20and%20Classification/Results/fps_throughput.jpeg)

## Improvements

Initially, the model used no pipeline and the total time taken by the model to process 900 frames was 350
seconds (about 6 minutes). After implementing a producer-consumer model which uses a queue to
transfer data from producer to consumer, all frames are getting processed in 250 seconds (about 4
minutes). The producer continuously reads data and tries to put it in the queue. The consumer
continuously reads the data from the queue and processes it. The producer-consumer model takes the
advantage of multithreading, where producer and consumer work in parallel, thus improving the
throughput. Below figure shows the FPS over processing 900 frames.

![Consumer Producer](https://github.com/tjbohari/Deep-Learning-using-Python/blob/main/Object%20Detection%20and%20Classification/Results/fps_throughput_consumer.jpeg)
