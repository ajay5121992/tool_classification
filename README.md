Tool Classification

Its an Image Classifier used for classifying the tool quality by  used in Mech. Industry.

tool_classification.py:
The Model has been tained using VGG16 arch. which is an benchmark for image classification, this arch. has also won the immage classifcation state of art in ILSVRC competition.


extra_ploating_images.py:
Before Training the data set of images has been extrapolated with ImageDataGenerator keras api, the extrapolation has been done by altering width,height,rotational angle,horizontal flip,vertical flip,shear. It is also noted that each differnet class has equal number imges for better classification and generialization for model
The images are resized (128,128,3) and scaled to 1./255

convert_to_tensorflow.py:
An Keras model has been generated from training which is then used for generating protobufs .pb file which is our tensorflow model used in android app to predict the captured photo among the certain classes.  

MainActvity.java:
The Main class of Android which runs and predicts the category of class.

activity_main.xml:
XML file for UI Design for caturing input from user, in which user clicks a button and results are diplayed on App 
