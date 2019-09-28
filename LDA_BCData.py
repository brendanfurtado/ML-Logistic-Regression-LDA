from prepareData import *
from LDA import *
import numpy as np
import pandas as pd
import time


bc_df = prepareData.dataframe
df = bc_df.copy()
del df['ID'] #Dropping an irrelevant feature that has nothing to do with the prediction of whether a tumor is benign or not


df['class_modified'] = pd.to_numeric((df['Class'] == 4)).astype(int)
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei']).astype(int)


#Standardize data
for column in df.columns[0:9]:
    df[column] = (df[column] - df[column].mean()) / df[column].std()


#LDA

LDA_BC = LDA()
df.insert(0, "Constant", 1)

df_copy = df.copy()
df_copy = df_copy.drop(columns=['Class'])

X = df_copy[df_copy.columns[0:10]]
Y = df_copy["class_modified"]


def k_fold_CV(data, model, k):

    all_data = data.iloc[np.random.permutation(len(data))]
    data_split = np.array_split(data, k)
    accuracies = np.ones(k)

    for i, data in enumerate(data_split):

        training_data = pd.concat([all_data, data_split[i], data_split[i]]).drop_duplicates(keep=False)

        model.fit(training_data[training_data.columns[0:10]], np.array(training_data[training_data.columns[10]]))

        prediction = model.predict(data_split[i][data_split[i].columns[0:10]], np.array(data_split[i][data_split[i].columns[10]]))

        accuracies[i] = model.evaluate_acc(data_split[i][data_split[i].columns[10]], prediction)

    return np.mean(accuracies)

#5-fold cross validation with LDA
start_time = time.time()
for i in range(0,5):
    start_iteration_time = time.time()
    results = k_fold_CV(df_copy, LDA_BC, 5)
    print(results)

    end_iteration_time = time.time()

    print("Run time for each 5-fold for LDA: %g" % (end_iteration_time-start_iteration_time))

end_time = time.time()

print("The average time for 5-fold cross validation was %g seconds" % ((end_time-start_time)/5.0))

#Results
# 0.973679690854444
# Run time for each 5-fold for LDA: 2.29762
# 0.973679690854444
# Run time for each 5-fold for LDA: 1.92426
# 0.973679690854444
# Run time for each 5-fold for LDA: 1.68653
# 0.973679690854444
# Run time for each 5-fold for LDA: 1.6405
# 0.973679690854444
# Run time for each 5-fold for LDA: 1.67886
# The average time for 5-fold cross validation was 1.84557 seconds










