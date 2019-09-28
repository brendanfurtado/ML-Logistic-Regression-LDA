from prepareData import *
from LogisticRegression import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#Further data manipulation before running the model

df = prepareData.dataframe.copy()

del df['ID'] #Dropping an irrelevant feature that has nothing to do with the prediction of whether a tumor is benign or not

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

#Used within k-fold cross validation to show how the cost (cross entropy) goes down after each iteration
#gradient descent
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


    return np.mean(accuracies), costList


#Experiments (logistic and breast cancer dataset)

#This part of the code changes to output different graphs, based on different alpha values

#Testing different learning rates
# alpha1 = .0005
# start_time = time.time()
#
#
# results1 = k_fold_CV(df_copy, lr_BC, 5, alpha1, 10)
# print(results1[0]) #Accuracy
# end_time = time.time()
# print("Elapsed time for Logistic Regression on the Breast Cancer set was %g seconds" % (end_time-start_time)) #result 1: 51.5082 seconds, result 2: 51.0199 seconds
#
#
# costList1 = results1[1]
#
# # Visualizing number of iterations vs Cost
# plt.plot(costList1, '-y', label='a = ' + str(alpha1))
# plt.xlabel("Iterations")
# plt.ylabel("Cost")
# plt.title("Number of Iterations vs Cost")
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()



#Average running time of 5-fold cross validation (5 times)
#Constant values alpha = .001, iterations per cross fold = 10 => 50 total iterations
start_time = time.time()
for i in range(0,5):
    start_iteration_time = time.time()
    results = k_fold_CV(df_copy, lr_BC, 5, .001, 20)
    print(results[0])

    end_iteration_time = time.time()

    print("Run time for each 5-fold: %g" % (end_iteration_time-start_iteration_time))

end_time = time.time()

print("The average time for 5-fold cross validation was %g seconds" % ((end_time-start_time)/5.0))
#Result 1: 47.3634 seconds
#Result 2: alpha = .0001
# 0.9707492486045514
# Run time for each 5-fold: 42.5362
# 0.9692893945899528
# Run time for each 5-fold: 44.7205
# 0.9692893945899528
# Run time for each 5-fold: 42.0015
# 0.9707492486045514
# Run time for each 5-fold: 43.5745
# 0.9707492486045514
# Run time for each 5-fold: 47.7812
# The average time for 5-fold cross validation was 44.1228 seconds

#Result 3: alpha = .001
# 0.9736904250751396
# Run time for each 5-fold: 42.3905
# 0.9707707170459425
# Run time for each 5-fold: 48.2474
# 0.9722305710605411
# Run time for each 5-fold: 43.5293
# 0.9707707170459425
# Run time for each 5-fold: 42.2336
# 0.969310863031344
# Run time for each 5-fold: 44.6118
# The average time for 5-fold cross validation was 44.2025 seconds


#Code to plot learning rate vs time to run
# times = [0,0,0,0,0,0,0]
# start = time.time()
# print(k_fold_CV(df_copy, lr_BC, 5, 0.00001, 10)) # 0.9692786603692571
# times[0] = time.time() - start
# start = time.time()
# print(k_fold_CV(df_copy, lr_BC, 5, 0.0001, 10)) # 0.9692786603692571
# times[1] = time.time() - start
# start = time.time()
# print(k_fold_CV(df_copy, lr_BC, 5, 0.001, 10)) # 0.9707492486045514
# times[2] = time.time() - start
# start = time.time()
# print(k_fold_CV(df_copy, lr_BC, 5, 0.01, 10)) # 0.9707707170459425
# times[3] = time.time() - start
# start = time.time()
# print(k_fold_CV(df_copy, lr_BC, 5, 0.05, 10)) # 0.9678510090167454
# times[4] = time.time() - start
# start = time.time()
# print(k_fold_CV(df_copy, lr_BC, 5, 0.1, 10)) # 0.9634714469729498
# times[5] = time.time() - start
# start = time.time()
# print(k_fold_CV(df_copy, lr_BC, 5, 0.5, 10)) # 0.9692571919278661
# times[6] = time.time() - start
#
# learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
# #times = [56.67494606971741, 58.5338819026947, 60.11971831321716, 58.255337953567505, 55.337260007858276, 49.810872077941895, 44.955501079559326]
# plt.plot(learning_rates, times, 'o')
# plt.xscale("log")
# plt.xlabel("log(Learning Rate)")
# plt.ylabel("Time (seconds)")
# plt.show()







