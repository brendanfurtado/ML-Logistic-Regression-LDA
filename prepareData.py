import pandas as pd


def createDataframe(file):

    desired_width = 320

    pd.set_option('display.width', desired_width)

    pd.set_option('display.max_columns', 11)

    dataset = pd.read_csv(file, header=None)

    dataset.columns = ['ID', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
                       'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
                       'Normal_Nucleoli', 'Mitoses', 'Class']
    return dataset

def scrubData(dataframe):

    # Any data with '?' will be removed
    dataframe = dataframe.drop([23, 40, 139, 145, 158, 164, 235, 249, 275, 292, 294, 297, 315,
                                321, 411, 617], axis=0)

    #Classifying the tumors (benign and malignant) as 0 and 1 respectively
    #And converting an object row to numerical
    dataframe['class_modified'] = pd.to_numeric((dataframe['Class'] == 4)).astype(int)
    dataframe['Bare_Nuclei'] = pd.to_numeric(dataframe['Bare_Nuclei']).astype(int)

    # Count of rows should reduce to 683 rows
    return dataframe



class prepareData:
   dataframe = createDataframe("breast-cancer-wisconsin.data")

   dataframe = scrubData(dataframe)

