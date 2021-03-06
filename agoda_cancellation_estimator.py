from __future__ import annotations
from typing import NoReturn

import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from IMLearn.base import BaseEstimator
import numpy as np
import pandas as pd
import datetime
import math


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters

        ----------


        Attributes
        ----------

        """
        self.estimator_ = KNeighborsClassifier(13)  # LinearRegression()
        self.fitted_ = False
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        data = AgodaCancellationEstimator.adjust_data(X)
        # y_adjusted = AgodaCancellationEstimator.adjust_response(y)
        y_adjusted = AgodaCancellationEstimator.adjust_response(X[:, 1], y)
        self.estimator_.fit(data, y_adjusted)
        self.fitted_ = True

    # @staticmethod
    # def adjust_response(X):
    #     y = np.zeros(X.shape[0])
    #     for sample in range(X.shape[0]):
    #         if isinstance(X[sample], str):
    #             y[sample] = 1
    #         else:
    #             y[sample] = 0
    #     return y

    @staticmethod
    def adjust_response(dates, y):
        new_y = np.zeros(y.shape[0])
        for sample in range(y.shape[0]):
            if not AgodaCancellationEstimator.isnat(y[sample]):
                new_y[sample] = (y[sample] - dates[sample]).days
            else:
                new_y[sample] = -1
        return new_y

    @staticmethod
    def adjust_true_labels(y, from_date, to_date):
        new_y = np.zeros(y.shape[0])
        for sample in range(y.shape[0]):
            if not AgodaCancellationEstimator.isnat(y[sample]):
                new_y[sample] = 0
            elif y[sample] >= from_date and y[sample] <= to_date:
                new_y[sample] = 1
            else:
                new_y[sample] = 0
        return new_y

    @staticmethod
    def predicted_cancellation_in_range(booking_date, days_to_cancel, from_date, to_date):
        cancellation_date = [booking_date[i] + np.timedelta64(int(days_to_cancel[i]), 'D') for i in
                             range(booking_date.shape[0])]
        prediction = []
        for i in range(booking_date.shape[0]):
            if (cancellation_date[i] >= from_date) and (cancellation_date[i] <= to_date):
                prediction.append(1)
            else:
                prediction.append(0)
        return np.array(prediction)

    @staticmethod
    def isnat(sample):
        nat_as_integer = np.datetime64('NAT').view('i8')
        dtype_string = str(sample.dtype)
        if 'datetime64' in dtype_string or 'timedelta64' in dtype_string:
            return sample.view('i8') == nat_as_integer
        return False  # it can't be a NaT if it's not a dateime

    @staticmethod
    def adjust_data(X):
        # days between booking and check in
        col_1 = AgodaCancellationEstimator.date_to_days(X[:, 0], X[:, 1])
        # days between checkout and check in
        col_2 = AgodaCancellationEstimator.date_to_days(X[:, 0], X[:, 2])
        # payment method(now/later)
        col_3 = AgodaCancellationEstimator.pay_Now_or_later(X[:, 3])
        # is first booking
        col_4 = AgodaCancellationEstimator.first_booking(X[:, 4])

        # not sure but made better score
        # col_5 = X[:, 5]

        # cancellation policy
        col_6 = AgodaCancellationEstimator.canellation_policy(X[:, 6])

        # col_7 = AgodaCancellationEstimator.first_booking(X[:, 7])
        # number of children
        # col_5 = X[:, 5]
        # number of adults
        # col_5 = X[:, 5]
        return np.array([col_1, col_2, col_3, col_4, col_6]).T

    @staticmethod
    def first_booking(X):
        false_scalar, true_scalar = 0, 1
        y = np.zeros(X.shape[0])
        for sample in range(X.shape[0]):
            if X[sample]:
                y[sample] = true_scalar
            else:
                y[sample] = false_scalar
        return y

    @staticmethod
    def canellation_policy(X):
        y = np.zeros(X.shape[0])
        for sample in range(X.shape[0]):
            if "100P" not in X[sample]:
                y[sample] = 100
            else:
                y[sample] = -1
        return y

    @staticmethod
    def pay_Now_or_later(X):
        now_scalar, later_scalar = -100, 100
        y = np.zeros(X.shape[0])
        for sample in range(X.shape[0]):
            if X[sample] == "Pay Now":
                y[sample] = now_scalar
            else:
                y[sample] = later_scalar
        return y

    @staticmethod
    def date_to_days(X1, X2):
        delta = abs(X1 - X2)
        return np.array([s.days for s in delta])

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        c = self.estimator_.predict(AgodaCancellationEstimator.adjust_data(X))
        return AgodaCancellationEstimator.predicted_cancellation_in_range(X[:, 0], c, np.datetime64('2018-12-04'),
                                                                          np.datetime64('2018-12-16'))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        predictions = self.predict(X)
        new_y= AgodaCancellationEstimator.adjust_true_labels(y, np.datetime64('2018-12-04'),np.datetime64('2018-12-16'))
        sum = 0
        for i in range(y.shape[0]):
            if predictions[i] == new_y[i]:
                sum += 1
        return sum / y.shape[0]
