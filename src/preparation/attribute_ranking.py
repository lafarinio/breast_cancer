import numpy as np

from src.shared.import_data import ImportData
from scipy import stats, array

def test_kolmogorovsmirnov():
    test_import = ImportData()
    column_names = np.array(['Clump Thickness', 'Class'])
    data = test_import.import_data(column_names)
    filteredFirstClass = list(filter(lambda a: a[1] == 2, data))
    filteredSecondClass = list(filter(lambda a: a[1] == 4, data))

    firstClassArray = array(filteredFirstClass)
    secondClassArray = array(filteredSecondClass)

    attribute1 = firstClassArray[:, 0]
    attribute2 = secondClassArray[:, 0]

    print(attribute1)
    print("-------")
    print(attribute2)
    print("------------------------")
    print(stats.ks_2samp(attribute1, attribute2))


if __name__ == "__main__":
    test_kolmogorovsmirnov()


