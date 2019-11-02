def input_train_data_folder():

    """ Default train folder is 'training_images_table2/' """
    """ Return trainining data images folder name"""

    train_data_folder = input('Please input train data folder name:')

    if train_data_folder == '':
        train_data_folder = 'training_images_table2/'

    return train_data_folder
