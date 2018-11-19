import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn import model_selection, tree
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import logging

log = logging.getLogger(__name__)


class CVClassifier(BaseEstimator):
    def __init__(self, intValue=0, stringParam="defaultValue", otherParam=None):
        super().__init__(self, intValue, stringParam, otherParam)

    def fit(self, X, y=None):
        raise NotImplementedError("You need to override fit()")

    def predict(self, X):
        raise NotImplementedError("You need to override predict()")

    def score(self, X, y=None):
        predictions = self.predict(X)
        mean_squared_error = np.mean((np.array(y) - np.array(predictions)) ** 2)
        return mean_squared_error

    def get_cv_score(self, X_train, y_train, num_folds):
        # See http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
        results = model_selection.cross_validate(self, X_train, y_train,
                                                 cv=num_folds,
                                                 return_train_score=True)

        # Pre-calculate common aggregates
        results['mean_test_score'] = np.mean(results['test_score'])
        results['mean_train_score'] = np.mean(results['train_score'])
        return results

    def __str__(self):
        return self.__class__.__name__


class SVMClassifier(CVClassifier):
    def __init__(self, kernel='linear', c_val=1):
        self.classifier = SVC(kernel=kernel, C=c_val)
        return

    def fit(self, X, y=None):
        self.classifier.fit(X, y)
        return

    def predict(self, X):
        return self.classifier.predict(X)


class NBClassifier(CVClassifier):
    def __init__(self):
        self.classifier = MultinomialNB()
        return

    def fit(self, X, y=None):
        self.classifier.fit(X, y)
        return

    def predict(self, X):
        return self.classifier.predict(X)


class NNClassifier(CVClassifier):
    """Superclass of a neural net regressor."""

    def __init__(self, tf_apply_func, name, training_rounds=5000,
                 learning_rate=5e-3):
        self.tf_apply_func = tf_apply_func
        self.name = name
        self.training_rounds = training_rounds
        self.learning_rate = learning_rate
        self.session = None
        self.in_ph = None
        self.out = None

    def fit(self, X, y=None):
        # log.debug("X Shape: {}, {} \n Y Shape: {}, {}".format(X.shape, X[0], y.shape, y[0]))

        if y is None:
            log.error("No labels input to fit!!")
            assert 0
        # Full reset betweeen fit() calls
        tf.reset_default_graph()
        self.loss = []

        self.in_ph = tf.placeholder(tf.float32, (None, X.shape[1]))
        out_ph = tf.placeholder(tf.int32, (None, y.shape[1]))
        self.out = self.tf_apply_func(self.in_ph, y.shape[1])
        # loss = tf.reduce_mean(tf.square(self.out - out_ph))
        # print(out_ph)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=out_ph,
                                                       logits=self.out))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        for i in range(self.training_rounds):
            r = self.session.run([train, loss, self.out],
                                 feed_dict={self.in_ph: X, out_ph: y})
            self.loss.append(r[1])
            # if i == self.training_rounds - 1:
            if not(i % 200):
                log.info("#  Training loss at round %s: %s" % (i, r[1]))
        plt.plot(self.loss)
        plt.show()
        return self

    def predict(self, X):
        result = self.session.run([self.out], feed_dict={self.in_ph: X})[0]
        # log.debug("Result Shape:{}".format(result.shape))
        # log.debug("Results: {}".format(result))
        argmax = np.argmax(result)
        # print("Argmax:{}".format(argmax))
        return argmax

    def __str__(self):
        return "NNSARSClassifier<%s>" % (self.name)


def neural_base(depth, hidden_units, fn, X, output_size):
    o = X
    for d_index in range(depth):
        o = tf.layers.dense(o, units=hidden_units, activation=fn)
    # No need for softmax layer if we use softmax_cross_entropy_with_logits_v2
    # it internally performs the softmax.
    # return tf.layers.dense(o, units=output_size, activation=tf.nn.softmax)
    return tf.layers.dense(o, units=output_size, activation=None)


def build_neural_classifier(rounds=20):
    classifier_list = []
    # for fn in [tf.nn.tanh, tf.nn.relu]:
    for fn in [tf.nn.relu]:
<<<<<<< Updated upstream
        # for hidden_units in [8, 16, 32]:
        for hidden_units in [8]:
            # for depth in [3, 5]:
=======
        #for hidden_units in [8, 16, 32]:
        for hidden_units in [8, 16]:
            #for depth in [3, 5]:
>>>>>>> Stashed changes
            for depth in [3]:
                # for learning_rate in [0.05, 0.005, 0.0005]:
                for learning_rate in [0.005]:
                    classifier_list.append(
                        [partial(neural_base, depth, hidden_units, fn),
                         "%s/%s/%s/%s" % (
                         depth, hidden_units, fn.__name__, str(learning_rate)),
                         learning_rate])
    return [NNClassifier(f, name, rounds, learning_rate) for
            f, name, learning_rate in classifier_list]
