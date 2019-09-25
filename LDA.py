import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as mp

# Linear discriminant analysis model
# Compute log-odds ratio
# Decision boundary, log-odds ratio>0, then output is 1; 0 otherwise

class LDA:


    # P(y=0)
    def probability_c1(self, n0, n1):
        return n1/(n0+n1)


    # P(y=1)
    def probability_c0(self, n0, n1):
        return n0/(n0+n1)


    # Mu1, if object from class1 contains feature xi, then i=1; 0 otherwise
    def mean_c1(self, X):
        sum_class1 = np.zeros(shape=(1, len(X[0])))
        for i in range(len(X)):
            sum_class1 += np.array(X[i])

        return sum_class1.T / len(X)


    # Mu0, if object from class0 contains feature xi, then i=1; 0 otherwise
    def mean_c0(self, X):
        sum_class0 = np.zeros(shape=(1, len(X[0])))
        for i in range(len(X)):
            sum_class0 += np.array(X[i])

        return sum_class0.T / len(X)


    # when class0
    def covariance_c0(self, X):
        sum0 = np.zeros(shape=(len(X[0]), len(X[0])))
        for i in range(len(X)):
            difference = np.array(X[i].T) - self.mean_c0(X)
            sum0 += (difference * difference.T)

        return sum0


    # when class1
    def covariance_c1(self, X):
        sum1 = np.zeros(shape=(len(X[0]), len(X[0])))
        for i in range(len(X)):
            difference = np.array(X[i].T) - self.mean_c1(X)
            sum1 += (difference * difference.T)

        return sum1


    # sum up two covariance
    def covariance(self, X1, X0):
        sum = np.zeros(shape=(len(X0[0]), len(X0[0])))
        covar0 = self.covariance_c0(X0)
        covar1 = self.covariance_c1(X1)

        sum = (covar0 + covar1)
        divide = (len(X1) + len(X0) - 2)
        covar = sum/divide

        return covar


    # computing log-odds ratio
    def fit(self, X, y):
        class1 = []
        class0 = []
        log_odds_ratio_list = np.zeros_like(y, float)

        for i in range(len(X.index)):
            if y[i] == 1:
                class1.append(X.iloc[i])
            elif y[i] == 0:
                class0.append(X.iloc[i])
        class1 = np.array(class1)
        class0 = np.array(class0)

        ratio = np.log(self.probability_c1(len(class0), len(class1))/self.probability_c0(len(class0), len(class1)))
        mu1 = self.mean_c1(class1)
        mu0 = self.mean_c0(class0)

        covar = self.covariance(class1, class0)
        in_covar = np.linalg.pinv(covar)
        c1 = np.dot(np.dot(mu1.T, in_covar), mu1)/2
        c0 = np.dot(np.dot(mu0.T, in_covar), mu0)/2
        w0 = ratio - c1 + c0

        for i in range(len(X.index)):
            xTw = np.dot(np.dot((X.iloc[i]), in_covar), (mu1 - mu0))
            log_odds_ratio_list[i] = w0 + xTw

        return log_odds_ratio_list


    # predicting single data point
    def predict(self, X, y):
        fit_list = self.fit(X, y)
        for i in range(len(X.index)):
            if fit_list[i] <= 0:
                fit_list[i] = 0
            else:
                fit_list[i] = 1

        return fit_list


    # accuracy function
    def evaluate_acc(self, y, prediction_y):

        y = list(y)

        for i in range(len(prediction_y)):

            differences = np.subtract(y, prediction_y)
            return (len(differences) - np.sum(np.abs(differences))) / len(differences)

wine_df = pd.read_csv("winequality-red.csv", delimiter=";")
wine_df["quality_modified"] = pd.to_numeric((wine_df["quality"] > 5) & (wine_df["quality"] < 11)).astype(int)

# Standardize Data
for column in wine_df.columns[0:11]:
    wine_df[column] = (wine_df[column] - wine_df[column].mean()) / wine_df[column].std()

# comparison_df = wine_df.groupby("quality_modified").mean()
# comparison_df.T.plot(kind="bar")
# plt.show()

# scatter_matrix(wine_df, alpha=0.3)
# plt.show()
lda = LDA()

wine_df.insert(0, "Constant", 1)

wine_df_copy = wine_df.copy()
wine_df_copy = wine_df_copy.drop(columns=["quality"])

X = wine_df[wine_df.columns[0:12]]
y = wine_df["quality_modified"]

def k_fold_CV(data, model, k):

    all_data = data.iloc[np.random.permutation(len(data))]
    data_split = np.array_split(data, k)
    accuracies = np.ones(k)

    for i, data in enumerate(data_split):

        training_data = pd.concat([all_data, data_split[i], data_split[i]]).drop_duplicates(keep=False)
        model.fit(training_data[training_data.columns[0:12]], np.array(training_data[training_data.columns[12]]))
        prediction = model.predict(data_split[i][data_split[i].columns[0:12]], np.array(data_split[i][data_split[i].columns[12]]))
        accuracies[i] = model.evaluate_acc(data_split[i][data_split[i].columns[12]], prediction)

    return np.mean(accuracies)


print(k_fold_CV(wine_df_copy, lda, 5))
