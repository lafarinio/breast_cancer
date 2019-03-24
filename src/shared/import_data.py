import pandas as pd
import numpy


class ImportData:
    def __init__(self,
                 # data_path='../../data/breast-cancer-wisconsin.data',
                 data_path='data/breast-cancer-wisconsin.data',
                 columns_path='data/breast-cancer-columns.names'):
        self.data_path = data_path
        self.columns_path = columns_path

    def import_data(self, selected_column_names=[]):
        columns_names = self.import_columns_names()
        if (selected_column_names == []):
            selected_column_names = columns_names
        mydata = pd.read_csv(self.data_path,
                             sep=',',
                             names=columns_names,
                             usecols=selected_column_names)
        return mydata

    def import_columns_names(self):
        columns_names = pd.read_csv(self.columns_path, sep=',', comment='#')
        columns_names = columns_names.to_numpy()
        return numpy.concatenate(columns_names, axis=0)
