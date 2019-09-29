import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import time
from LDA import *

wine_df = pd.read_csv("winequality-red.csv", delimiter=";")
wine_df["quality_modified"] = pd.to_numeric((wine_df["quality"] > 5) & (wine_df["quality"] < 11)).astype(int)

# Analyze distribution of wine data frame features
for i in wine_df.columns:
    print(wine_df[i].describe())


## Graphs mean feature value comparison of different classes
comparison_df = wine_df.groupby("quality_modified").mean()
comparison_df[comparison_df.columns[0:6]].T.plot(kind="bar")
plt.xlabel("Feature")
plt.ylabel("Average Value across Class")
plt.xticks(fontsize=6, rotation=0)
plt.show()
comparison_df[comparison_df.columns[6:14]].T.plot(kind="bar")
plt.xlabel("Feature")
plt.ylabel("Value")
plt.xticks(fontsize=6, rotation=0)
plt.show()



# Standardize Data
for column in wine_df.columns[0:11]:
    wine_df[column] = (wine_df[column] - wine_df[column].mean()) / wine_df[column].std()

# Makes scatter matrix of proper columns
sm = scatter_matrix(wine_df[wine_df.columns[[0,3,6,8,9,10]]])

for ax in sm.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 5, rotation = 0)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 5, rotation = 90)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

wine_df.insert(0, "Constant", 1)
wine_df.insert(12, "Interaction 2", wine_df["total sulfur dioxide"]*wine_df["sulphates"])

wine_df_copy = wine_df.copy()
wine_df_copy = wine_df_copy.drop(columns=["quality"])


lda = LDA()

# Test times for LDA model with wine dataset
start = time.time()
times = [0,0,0,0,0]
for i in range(5):
    res = k_fold_CV(wine_df_copy, lda, 5) # 0.737337382445141
    print(res)
    times[i] = float(time.time()-start)
    start = time.time()

print(np.mean(np.array(times)))


from LR import *
lr_wine = LogisticRegression(np.zeros((1, 13), float))
# Measure run time vs learning rate
times = [0,0,0,0,0,0,0]
start = time.time()
print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.00001, 10)) # 0.7191927899686521
times[0] = time.time() - start
start = time.time()
print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.0001, 10)) # 0.7379584639498432
times[1] = time.time() - start
start = time.time()
print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.001, 10)) # 0.751716300940439
times[2] = time.time() - start
start = time.time()
print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.01, 10)) # 0.6666457680250784
times[3] = time.time() - start
start = time.time()
print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.05, 10)) # 0.6522923197492163
times[4] = time.time() - start
start = time.time()
print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.1, 10)) # 0.6641438087774294
times[5] = time.time() - start
start = time.time()
print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.5, 10)) # 0.6522903605015673
times[6] = time.time() - start

## Plot learning rate with time
learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
times = [56.67494606971741, 58.5338819026947, 60.11971831321716, 58.255337953567505, 55.337260007858276, 49.810872077941895, 44.955501079559326]
plt.plot(learning_rates, times, 'o')
plt.xscale("log")
plt.xlabel("log(Learning Rate)")
plt.ylabel("Time (seconds)")
plt.show()



print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.001, 10)) # 0.7504663009404389 - with second interaction term
print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.001, 15)) # 0.7517202194357366 - with second interaction term
print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.001, 10)) # 0.7523491379310345 - with second interaction term
print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.01, 10)) # 0.6666457680250784 - with second interaction term
print(k_fold_CV(wine_df_copy, lr_wine, 5, 0.1, 10)) # 0.6504173197492162 - with second interaction term