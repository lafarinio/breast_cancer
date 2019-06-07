import pandas as pd
import numpy
import definitions


class ImportData:
    def __init__(self):
        self.data_path = definitions.BC_DATA_PATH
        self.columns_path = definitions.BC_COLUMNS_NAMES_PATH

    def import_all_data(self) -> numpy.ndarray:

        columns_names = self.import_columns_names()
        usecols = self.import_columns_names_without_class()

        mydata = pd.read_csv(self.data_path,
                             sep=',',
                             index_col=0,
                             names=columns_names,
                             usecols=usecols)
        return mydata.values

    def import_data(self, selected_column_names: numpy.ndarray) -> numpy.ndarray:

        columns_names = self.import_columns_names()

        mydata = pd.read_csv(self.data_path,
                             sep=',',
                             names=columns_names,
                             usecols=selected_column_names)

        return mydata.values

    def import_columns_names(self) -> numpy.ndarray:
        columns_names = pd.read_csv(self.columns_path, sep=',', comment='#', header=None)
        columns_names = columns_names.to_numpy()
        return numpy.concatenate(columns_names, axis=0)

    def import_columns_names_without_class(self) -> numpy.ndarray:
        columns_names = self.import_columns_names()
        indices = range(0,10)
        result = numpy.take(columns_names, indices)

        return result

    def cut_columns_from_data(self, columns: []) -> numpy.ndarray:
        columns_names = self.import_columns_names()
        usecols = self.import_columns_names_without_class()
        print(columns)
        mydata = pd.read_csv(self.data_path,
                             sep=',',
                             index_col=0,
                             names=columns_names,
                             usecols=usecols)
        tmp = mydata.drop(columns=columns)
        print(tmp)
        return tmp.values
