from prepareData import *
from LogisticRegression import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp


bc_df = prepareData.dataframe
df = bc_df.copy()
del df['ID'] #Dropping an irrelevant feature that has nothing to do with the prediction of whether a tumor is benign or not

df['class_modified'] = pd.to_numeric((df['Class'] == 4)).astype(int)
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei']).astype(int)


#Standardize data
for column in df.columns[0:9]:
    df[column] = (df[column] - df[column].mean()) / df[column].std()


#Logistic regression

lr_BC = LogisticRegression(np.zeros((1,10), float))
df.insert(0, "Constant", 1)

df_copy = df.copy()
df_copy = df_copy.drop(columns=['Class'])

X = df_copy[df_copy.columns[0:10]]

Y = df_copy["class_modified"]

def calculateCost(model, costList):

    d1 = np.concatenate(model[1], axis=0)

    j = 0
    for elements in (model[1]):
        costList.append(d1[j])
        j += 1

    return costList


def k_fold_CV(data, model, k, learning_rate, iteration):

    all_data = data.iloc[np.random.permutation(len(data))]
    data_split = np.array_split(data, k)
    accuracies = np.ones(k)
    costList = []
    for i, data in enumerate(data_split):

        training_data = pd.concat([all_data, data_split[i], data_split[i]]).drop_duplicates(keep=False)
        model_copy = model.fit(training_data[training_data.columns[0:10]], np.array(training_data[training_data.columns[10]]),
                  learning_rate=learning_rate, iteration=iteration)

        costList = calculateCost(model_copy, costList)

        prediction = model.predict(data_split[i][data_split[i].columns[0:10]])
        accuracies[i] = model.evaluate_acc(data_split[i][data_split[i].columns[10]], prediction)

    #Visualizing number of iterations vs Cost
    plt.plot(costList)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Number of Iterations vs Cost")
    plt.show()

    return np.mean(accuracies)

print(k_fold_CV(df_copy, lr_BC, 5, 0.001, 20))


#print(k_fold_CV(df_copy, lr_BC, 5, 0.05, 10))
#print(k_fold_CV(df_copy, lr_BC, 5, 0.01, 10))