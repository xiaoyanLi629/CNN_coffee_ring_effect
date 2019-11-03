# CNN_coffee_ring_effect
This project uses CNN model to analyze water samples of coffee ring effect

Dataset:

The csv file cluster_chemistry_result.csv is the water samples cluster analysis result.

Two set of images collected from same 30 water samples on two different days.
First set images is 150 images consisting of 5 replicates of 30 water samples on 6/13/2017.
Second set of images is 150 images consisting of 5 replicates of 30 water samples on 8/14/2017.
Two sets of datasets created by these collected 300 images.

The first dataset is training dataset (training_raw folder) consisting of 180 images. Consisting three replicates of each water sample images in 6/13/2017 dataset and three replicates of each water sample images in 6/13/2019 dataset.

The Second dataset is testing dataset (testing_raw folder) consisting of 180 images. Consisting two replicates of each water sample images in 8/14/2017 dataset and two replicates of each water sample images in 8/14/2017 dataset.

The training datset and testing dataset were created by ramdomly selecting images from 6/13/2017 and 8/14/2017 images.

Preprocessing images:

RGB_noemalize.m
  This file normalize images colors to smooth images and the smoothed images increased accuracy.
  training_raw folder images were normalized to training folder.
  testing_raw folder images were normalized to testing folder.

1, read_images_to_pkl.py
  Independent file
    This file read images from folder and converted images from RGB to black and white and save all the images in one pkl file.
    
Function files:
Under CNN_main.py file

1, train_folder.py
  Provide the training folder name
    
2, test_folder.py
    Provide the testing folder name
    
3, load_train_data.py
  This files reads images pkl file and return training data, training data labels in torch format
    
4, load_test_data.py
  This files reads images pkl file and return training data, training data labels in torch format
 
5, test_images.py
  This file reads images data and use CNN model to predict the results and compare the predicted results with its true label
  This file returns tested samples accuracy, prediction class and total loss
    
7, model_result.py
  Read model name, images true class label, images number, images predicted class and compare the predicted class and true class
  Return a list of misclssified images file code
    
8, missclassify_class.py
  Read images true class label, images number, images predicted class and compare the predicted class and true class
  Return a list of misclssified images file code
  This is part of function 7

Result analysis files:

1, Each_run_accuracy_box_plot.py
  This files reads all 10 models accuracy results and plots each models' last 100 accuracy results with scale bars
  Output figure file is Test accuracy figure.jpg
  
2, Each_class_accuracy_plot.py
  This file reads all 10 models accuracy results and plots accuracy of each class (six water samples classes in total) of the last 100 models of total ten runs.
  Output file is Test accuracy of each class.jpg
  
3, classification_result_analysis.py
  This file reads all 10 models accuracy results and plots three figures.
   1, accuracy of each class (six water samples classes in total) of the last 100 models of total ten runs. This is the same as file Each_class_accuracy_plot.py
   Output file is Test accuracy of each class.jpg
   2, accuracy of each image of total ten runs and last 100 models. Ordered in class number.
   Output file is Mis-classification percentage color class.jpg
   3, accuracy of each image of total ten runs and last 100 models zero mis-classification images. Ordered in class number.
   Output file is Mis-classification percentage color class without zero mis-classification.jpg
   
4, testing_accP_plot.py
  This file reads accuracy result csv file of each run. Then plots the accuracy of each run of the last 200 iteration models.
  Output file is Mis-classification percentage.jpg
  
5, mian_test.py
  This file reads model weighs file and test testing files by the model. Output confusion matrix of the testing images.
  10 confusion matrix file combined together by online kit
Files running flow:

1, run the CNN_main.py first. The file includes the model. read data and train the model by training data. Save 10 runs.

2, run Each_run_accuracy_box_plot.py

3, run Each_class_accuracy_plot.py

4, classification_result_analysis.py

5, testing_accP_plot.py

6, mian_test.py
