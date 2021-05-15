# Hand_Gesture_Recognition_through_CNN

### Introduction
The project aims at recognition of alphabets communicated through hand gestures in an image as well as a live video stream. Hand Gesture Recognition pipeline includes background 
subtraction to extract hand from image/video frame by extracting skin colour in image using OpenCV and Python. Once background subtraction is done recognition/classification task
is done using a custom CNN architecture. The model is able to recognize all the 26 English alphabets in image as well as video.

##### American Sign Language
![alt text](https://github.com/Yuvnish017/Hand_Gesture_Recognition_through_CNN/blob/master/American_Sign_Language.png?raw=true)

### Prerequisites
- Python
- OpenCV
- Keras
- Numpy
- CSV

### Contents
- hand_gesture_recognition.py
  > Python file that loads dataset and defines CNN model architecture and responsible for training the model

- hand_gesture_recognition.ipynb<br />
  > Google colab notebook version of hand_gesture_recognition.py file
  
- image_to_csv.py<br />
  > Python file for converting the dataset containing images in .jpeg format into a single .csv file.
  
- testing_images.py<br />
  > Inference code for testing the model's performance on single input image.
  
- live_video_stream_test.py<br />
  > Inference code for testing model's performance on video.
  
- testing_images folder<br />
  > Cotains images of all 26 alphabets (one image per one alphabet) for testing

### Dataset
Dataset can be downloaded in .csv format and original images also from this link - [Download Link](https://drive.google.com/drive/folders/1k0dZLOlwXsbJN6x9IHTFzdG3KAUYLs0F?usp=sharing)
