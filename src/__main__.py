from src.shared.import_data import ImportData

# package which was told to use:
# https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.ks_2samp.html


def main(args=None):

    test_import = ImportData()
    data = test_import.import_data()
    print("This is the main routine.")
    print(data)


if __name__ == "__main__":
    main()
