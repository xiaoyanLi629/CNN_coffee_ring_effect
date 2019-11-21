# CNN_coffee_ring_effect
This project uses a CNN model to analyze images of tap water fingerprints, assigning them to groups with similar water chemistry.

Dataset:

cluster_chemistry_result.csv 
  Thirty tap water samples were collected from different cities. The water chemistry in each tap water sample was evaluated using standard methods. Cluster anlaysis was used to assign tap water samples into groups with similar water chemistry. The csv file cluster_chemistry_result.csv is the result of the cluster analysis, showing which tap water samples have similar water chemistry. 


Raw image files (training_raw and testing_raw folders)
   Each tap water sample was dried on an aluminum slide, utilizing the coffee ring effect. Images were collected of the coffee-ring pattern for each tap water sample. Two set of images were collected from the thirty water samples on two different days. The first set images consisted of 5 replicates of the 30 water samples, for a total of 150 images. The second set of images consisted of 5 replicates of the 30 water samples collected on a second day, for a total of 150 more images. Overall there were 300 collected images. These 300 images were alloted into a training dataset and a testing dataset. 
   
   The training datset (training_raw folder) and testing dataset (testing_raw folder) were created by randomly selecting an equal number of images for each sample from the images analyzed on the first day and from the images analyzed on the second day. The training dataset contained 180 images, including three replicates from each tap water sample from the images collected on the first day and three replicates from each tap water sample collected on the second day. The testing dataset contained of 120 images, including two replicates from each tap water sample from the images collected on the first day and two replicates from each tap water sample collected on the second day. 
 

Preprocessed images:

  RGB_noemalize.m
    This code normalizes image colors to smooth images. 
    training_raw folder images were normalized and saved to the training folder. 
    testing_raw folder images were normalized and saved to the testing folder.
  
  read_images_to_pkl.py
    This file reads images from the folder, 
    converts images from RGB to black and white, and 
    saves all the images in one pkl file.
    
    
Function files can be found under the CNN_main.py file

  1) train_folder.py
      Provides the training folder name.
    
  2) test_folder.py
      Provides the testing folder name

  3) load_train_data.py
      This file reads the pkl file of the training images and returns training data and training data labels in torch format.

  4) load_test_data.py
      This file reads the pkl file of the testing images and returns the testing data and testing data labels in torch format.

  5) test_images.py
      This file reads the images, uses the CNN model to classify the images, and compares the CNN classification with the group assigned by cluster analysis of the water chemistry data. This file returns the CNN model classification, the accuracy of the CNN model in classifying the images into the group that was assigned by cluster analysis of the water chemistry data, and the total loss. 

  6) model_result.py
      This file reads the model name, the classification assigned by cluser analysis of the water chemistry data, the image identification number, and the classification assigned by the CNN model. It also compares the classification by CNN vs cluster anlaysis of water chemistry. The file returns a list of misclassified images.

  7) missclassify_class.py
      This file reads the classification assigned by the cluster analysis of water chemistry data, the image identification number, and the classification assigned by the CNN model. It also compares the classification by CNN vs cluster anlaysis of water chemistry. The file returns a list of misclassified images. This is part of function 6.


Result analysis files:

1) Each_run_accuracy_box_plot.py
    This file reads the accuracy results for 10 independently trained CNN models and plots each models' last 100 accuracy results with scale bars. The output figure file name is "Test accuracy figure.jpg"
  
2) Each_class_accuracy_plot.py
  This file reads the accuracy results for 10 independently trained CNN models and plots the accuracy of each class for the last 100 models. The output file is named "Test accuracy of each class.jpg"
  
3) classification_result_analysis.py
    This file reads the accuracy results for 10 independently trained CNN models and plots three figures. First it plots the accuracy of the classification for each class for the last 100 models, using the same procedure as used in "Each_class_accuracy_plot.py". The output file is named "Test accuracy of each class.jpg". Second it plots the accuracy of the classification for each image for the last 100 models of 10 independently trained CNN models, ordered by class number. The output file is named "Mis-classification percentage color class.jpg". Third it plots the accuracy of the classification for the zero mis-classification images for the last 100 models of 10 independently trained CNN models, ordered by class number. The output file is named "Mis-classification percentage color class without zero mis-classification.jpg".
   
4) testing_acc_plot.py
    This file reads the accuracy result csv file of each run, and then plots the accuracy of each run for the last 200 iteration models. The output file is named "Mis-classification percentage.jpg".
  
5) mian_test.py
    This file reads the model weights file and testing files by the model. The output is a confusion matrix of the testing images. The confusion matrix file is combined together by an online kit.



Work flow:

1) Run the CNN_main.py first. The file includes the CNN model. Read the data and train the model using the training dataset. Save 10 runs.

2) Run Each_run_accuracy_box_plot.py

3) Run Each_class_accuracy_plot.py

4) Run classification_result_analysis.py

5) Run testing_acc_plot.py

6) Run mian_test.py
