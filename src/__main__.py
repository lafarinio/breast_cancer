import numpy as np

from src.shared.import_data import ImportData
from scipy import stats, array


# package which was told to use:
# https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.ks_2samp.html

def testKolmogorovSmirnov(selected_column_name):
    test_import = ImportData()
    column_names = np.array([selected_column_name, 'Class'])
    data = test_import.import_data(column_names)
    filteredFirstClass = list(filter(lambda a: a[1] == 2, data))
    filteredSecondClass = list(filter(lambda a: a[1] == 4, data))

    firstClassArray = array(filteredFirstClass)
    secondClassArray = array(filteredSecondClass)

    attribute1 = firstClassArray[:, 0]
    attribute2 = secondClassArray[:, 0]

    sortedAttribute1 = sorted(attribute1)
    sortedAttribute2 = sorted(attribute2)

    norm1 = stats.norm.pdf(sortedAttribute1, np.mean(sortedAttribute1), np.std(sortedAttribute1))
    norm2 = stats.norm.pdf(sortedAttribute2, np.mean(sortedAttribute2), np.std(sortedAttribute2))
    print(norm1)
    print("-------")
    print(norm2)
    print("------------------------")
    print(stats.ks_2samp(norm1, norm2))


def main(args=None):
        testKolmogorovSmirnov('Mitoses')


if __name__ == "__main__":
    main()

