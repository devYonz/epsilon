import numpy as np

from ml.classifier import Classifier


class LogisticClassifier(Classifier):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run solver to fit a logistic regression model model.

        :param x: Training example inputs. Shape (m, n).
        :param y: Training example labels. Shape (m,).
        :return:
        """
        # Design x shape has samples on rows
        (m, n) = np.shape(x)
        # Support for theta 0 + theta1* x1  : [1 | x^i]
        # Randomly initialize the starting theta
        theta = np.zeros(n)
        diff = 1
        alpha = 0.2
        iteration = 1
        while diff > 1E-5 and iteration < 2000:
            iteration += 1
            theta_old = theta
            z = np.dot(x, theta)
            # TODO Below should probably be removed
            h_x = self.sigmoid(z)
            # Log likely hood equation to be maximized
            # l_theta = (y * np.log(h_x) + (1 - y) * np.log(1 - h_x)).sum()
            gradient = 1/m * x.T @ (y - h_x)
            theta = theta_old + alpha * gradient  # Maximize log likely hood
            diff = np.linalg.norm(theta_old-theta)
            # *** END CODE HERE ***
        self.theta = theta
        return self.theta

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return self.sigmoid((x @ self.theta))

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))
