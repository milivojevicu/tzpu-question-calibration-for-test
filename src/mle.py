import numpy as np
import math

EPS = 0.000000000001


class MLE:
    def __init__(self, X, Y):
        self._num_q = Y.shape[1]
        self._num_examinees = Y.shape[0]
        param = [np.random.uniform(low=0, high=3, size=(self._num_q)),
                 np.random.uniform(low=-3, high=3, size=(self._num_q)),
                 np.random.uniform(low=0, high=1, size=(self._num_q))]
        self._params = np.array(param).T.reshape(self._num_q, 3)  # shape=(num_questions, 3)
        self._X = X  # tetha | shape=(num_examinees)
        self._Y = Y  # ground truth | shape=(num_questions, num_examinees)
        self._lr = 0.005

    def cost(self, pred):
        error = np.zeros(self._num_q)

        for i in range(self._num_q):
            Y = self._Y.T
            for y, y_hat in zip(Y[i], pred[i]):
                error[i] += y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat)

        return -error

    def predict_all(self):
        pred = np.zeros((self._num_q, self._num_examinees))

        for i in range(self._num_q):
            for j in range(self._num_examinees):
                a, b, c = self._params[i]
                t = self._X[j]

                y_hat = c + (1 - c) / math.exp(-a * (t - b))

                if y_hat <= 0.5:
                    y_hat = EPS
                elif y_hat > 0.5:
                    y_hat = 1 - EPS

                pred[i][j] = y_hat

        return pred

    def calc_gradients(self, pred, error):
        grad = np.zeros((self._num_q, 3))

        for i in range(self._num_q):
            dE = np.zeros(3)

            for j in range(self._num_examinees):
                y = self._Y[j][i]
                y_hat = pred[i][j]
                a, b, c = self._params[i]
                t = self._X[j]

                dE_dy = (1 - y) / (1 - y_hat) - y / y_hat
                dy_dz = (1 - c) * math.exp(-(a * (t - b))) / (1 + math.exp(-(a * (t - b))) ** 2)
                dE_dz = dE_dy * dy_dz

                dE_da = dE_dz * (self._X[j] - b)
                dE_db = dE_dz * (-a)
                dE_dc = dE_dy / 2

                dE[0] += dE_da
                dE[1] += dE_db
                dE[2] += dE_dc

            dE /= self._num_examinees
            grad[i] = dE

        return grad

    def calc_gradients2(self, pred, error):
        grad = np.zeros((self._num_q, 3))

        for i in range(self._num_q):
            dE = np.zeros(3)

            for j in range(self._num_examinees):
                y = self._Y[j][i]
                y_hat = pred[i][j]
                a, b, c = self._params[i]
                t = self._X[j]

                dE_da = -math.exp(-a * (t - b)) * (1 - c) * (b - t) / (math.exp(-a * (t - b)) + 1) ** 2
                dE_db = -a * math.exp(-a * (t - b)) * (1 - c) / (math.exp(-a * (t - b)) + 1) ** 2
                dE_dc = 1 - 1 / (1 + math.exp(-a * (t - b)))

                dE[0] += dE_da
                dE[1] += dE_db
                dE[2] += dE_dc

            dE /= self._num_examinees
            grad[i] = dE

        return grad

    def accuracy(self, pred):
        acc = 0
        a = 0
        for i in range(self._num_q):
            for j in range(self._num_examinees):
                y = self._Y[j][i]
                a += pred[i][j]

                if pred[i][j] < 0.5 and y == 0:
                    acc += 1 - pred[i][j]  # replace with 1
                if pred[i][j] > 0.5 and y == 0:
                    acc += pred[i][j]  # replace with 1

        acc /= self._num_q * self._num_examinees
        print('acc: ', acc, a)
        return acc

    def fit(self, epochs=5000):
        for i in range(epochs):
            pred = self.predict_all()
            error = self.cost(pred)
            grad = self.calc_gradients(pred, error)

            self._params -= self._lr * grad

            for j in range(self._params.shape[0]):
                if self._params[j][0] <= 0:
                    self._params[j][0] = EPS
                elif self._params[j][0] >= 3:
                    self._params[j][0] = 3 - EPS

                if self._params[j][1] <= -3:
                    self._params[j][1] = EPS
                elif self._params[j][1] >= 3:
                    self._params[j][1] = 3 - EPS

                if self._params[j][2] <= 0:
                    self._params[j][2] = 0 + EPS
                elif self._params[j][2] >= 1:
                    self._params[j][2] = 1 - EPS

            if i % 100 == 1:
                self._lr /= 2
                print('Cost values:\n')
                print(self._params)
                print(error)
                self.accuracy(pred)
