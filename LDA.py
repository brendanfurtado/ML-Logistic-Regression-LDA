import numpy as np
import pandas as pd

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

        return sum_class1 / len(X)


    # Mu0, if object from class0 contains feature xi, then i=1; 0 otherwise
    def mean_c0(self, X):
        sum_class0 = np.zeros(shape=(1, len(X[0])))
        for i in range(len(X)):
            sum_class0 += np.array(X[i])

        return sum_class0 / len(X)


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

        y1 = self.probability_c1(len(class0), len(class1))
        y0 = self.probability_c0(len(class0), len(class1))
        ratio = np.log(y1 / y0)
        mu1 = self.mean_c1(class1)
        mu1 = mu1.T
        mu0 = self.mean_c0(class0)
        mu0 = mu0.T

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


    def confusion_matrix(self, y, prediction_y):
        matrix = np.zeros(shape=(2, 2))
        y = np.array(y)
        prediction_y = np.array(prediction_y)

        for i in range(len(prediction_y)):
            if y[i] == 1:
                if prediction_y[i] != 1:
                    matrix[0][1] = matrix[0][1] + 1
                else:
                    matrix[0][0] = matrix[0][0] + 1
            elif y[i] == 0:
                if prediction_y[i] != 0:
                    matrix[1][0] = matrix[1][0] + 1
                else:
                    matrix[1][1] = matrix[1][1] + 1

        return matrix

def k_fold_CV(data, model, k):

    all_data = data.iloc[np.random.permutation(len(data))]
    data_split = np.array_split(data, k)
    accuracies = np.ones(k)
    matrix = np.zeros(shape=(2, 2))

    for i, data in enumerate(data_split):

        training_data = pd.concat([all_data, data_split[i], data_split[i]]).drop_duplicates(keep=False)
        model.fit(training_data[training_data.columns[0:len(data.columns)-1]], np.array(training_data[training_data.columns[len(data.columns)-1]]))
        prediction = model.predict(data_split[i][data_split[i].columns[0:len(data.columns)-1]], np.array(data_split[i][data_split[i].columns[len(data.columns)-1]]))
        accuracies[i] = model.evaluate_acc(data_split[i][data_split[i].columns[len(data.columns) - 1]], prediction)
        matrix += model.confusion_matrix(data_split[i][data_split[i].columns[len(data.columns) - 1]], prediction)

    return np.mean(accuracies), matrix
