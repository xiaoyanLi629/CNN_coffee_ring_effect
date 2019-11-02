def input_test_data_folder():

    """ Default train folder is 'testing_images_table2/' """
    """ Return testing data images folder name"""

    test_data_folder = input('Please input test data folder name:')

    if test_data_folder == '':
        test_data_folder = 'testing_images_table2/'

    return test_data_folder
