'''
Base IO class for all datasets
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import abc


class dataset:
    """ 
    dataset: Abstract Class 
    Entries: dataset_name: the name of the dataset
             dataset_description: the textual description of the dataset
    """

    data = None
    dataset_name = None
    dataset_description = None
    dataset_source_file_path = None
    
    # Initialization function
    def __init__(self, dName=None, dDescription=None, dPath=None):
        '''
        Parameters: dataset name: dName, dataset description: dDescription
        Assign the parameters to the entries of the base class
        '''
        self.dataset_name = dName
        self.dataset_description = dDescription
        self.dataset_source_file_path = dPath
    
    # Information print function
    def print_dataset_information(self):
        '''
        Print the basic information about the dataset class
        including the dataset name, and dataset description
        '''
        print('Dataset Name: ' + self.dataset_name)
        print('Dataset Description: ' + self.dataset_description)

    # Dataset load abstract function
    @abc.abstractmethod
    def load(self, train):
        return
    
    