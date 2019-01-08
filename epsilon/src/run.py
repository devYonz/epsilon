import click
from functools import partial
import json
import logging
import numpy as np
import os
import sys
import tensorflow as tf

from ml.feature import Feature
from ml.logistic import LogisticClassifier
from classifiers import NBClassifier, NNClassifier, SVMClassifier
from processor import get_text_body, create_dictionary, transform_text
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[handler])
log = logging.getLogger(__name__)

# log.setLevel(logging.DEBUG)
PROJECT_KEY = 'projects'
ISSUES_KEY = 'issues'
ASSIGNEE_KEY = 'assignee'
NOTES_FILE = 'notes.log'
COUNTER_FILE = 'counts.log'
TOTAL_COUNT = 'total_count'
ASSIGNEE_COUNT = 'assignee_count'
DUPLICATE_COUNTS = 'duplicate_counts'
DUPLICATE_PERCENT = 'duplicate_percent'
ISSUES_DEDUP = {}
ASSIGNEE_DEDUP = {}


def get_stopwords():
    stopwords = []
    stopwords_path = os.path.join('datasets', 'stop_words.txt')
    if os.path.exists(stopwords_path):
        with open(stopwords_path, 'r') as stopwords_file:
            stopwords = stopwords_file.readlines()
    return set(stopwords)


def neural_base(depth, hidden_units, fn, X, output_size):
    o = X
    for d_index in range(depth):
        o = tf.layers.dense(o, units=hidden_units, activation=fn)
    # No need for softmax layer if we use softmax_cross_entropy_with_logits_v2
    # it internally performs the softmax.
    # return tf.layers.dense(o, units=output_size, activation=tf.nn.softmax)
    return tf.layers.dense(o, units=output_size, activation=None)


def build_neural_classifier(rounds=20, prefix='E'):
    classifier_list = []
    for fn in [tf.nn.tanh, tf.nn.relu]:
        # for fn in [tf.nn.relu]:
        for hidden_units in [8, 16, 32]:
            # for hidden_units in [8]:
            for depth in [3, 5]:
                # for depth in [3]:
                for learning_rate in [0.5, 0.05, 0.005]:
                    # for learning_rate in [0.005]:
                    classifier_list.append(
                        [partial(neural_base, depth, hidden_units, fn),
                         "%s%s-%s.%s.%s" % (prefix, depth, hidden_units, fn.__name__, str(learning_rate)),
                         learning_rate])
    return [NNClassifier(f, name, rounds, learning_rate) for f, name, learning_rate in classifier_list]


def take_note(text, directory):
    with open(os.path.join(directory, NOTES_FILE), 'a') as f:
        f.write(text)
    f.close()


def vectorize_json(dataset, directory):
    """ Returns a vector representation of the JIRA ticket referenced
    Ignores stopwords in object
     Adds keywords regardless of frequency

    :param json_ticket:
    :return:
    """
    count = 0
    assignee_count = 0

    issue_features = Feature()
    issues_lines = []
    assignees = []
    stopwords = get_stopwords()

    # Handle JIRA datasets with project breakouts
    issues = []
    if type(dataset) == dict and PROJECT_KEY in dataset:
        for project in dataset.get(PROJECT_KEY):
            issues.extend(project.get(ISSUES_KEY))
    else:
        issues = dataset

    # For a dataset of jira ticekts build matrix representation
    for i, issue in enumerate(issues):
        if i % 1000 == 0:
            log.debug(f'Processing ticket #{i}')

        if not ISSUES_DEDUP.get(issue.get('key')):
            count += 1
            ISSUES_DEDUP[issue.get('key')] = True

        words = get_text_body(issue, stopwords)
        issues_lines.append(words)
        assignee = issue.get(ASSIGNEE_KEY)
        assignees.append(assignee)

    # Construct the words into vector of >5 occurrence
    log.debug(f'Completed processing tickets  ticket, building dictionary')
    dictionary = create_dictionary(issues_lines)
    log.debug(f'Completed building dictionary, vectorizing tickets')
    train_matrix = transform_text(issues_lines, dictionary)

    # Setup class Label ENUMS: Y label assignees -> int class number
    for class_name in list(set(assignees)):
        issue_features.set_target_id(class_name)
        assignee_count += 1
    assignee_vector = []
    for assignee in assignees:
        assignee_vector.append(issue_features.get_target_id(assignee))

    # Build the train feature vector
    issue_features.data = train_matrix
    issue_features.target = assignee_vector
    train_labels = issue_features.target

    duplicates = len(issues) - count
    dup_percent = 100 * duplicates / len(issues)
    log.debug(f'Ticket count:{count}  Duplicates:{duplicates} or {dup_percent}%')

    # Output global counts and stats
    # Count total number of issues, unique assignees, unique tickets
    counts = {TOTAL_COUNT: count,
              ASSIGNEE_KEY: assignee_count,
              DUPLICATE_COUNTS: duplicates,
              DUPLICATE_PERCENT: dup_percent}

    # Store counts
    with open(os.path.join(directory, COUNTER_FILE), 'w') as f_counter:
        json.dump(counts, f_counter)

    return train_matrix, train_labels


def production_neural_classifer(rounds=200, prefix='E'):
    """ The function returns the DNN architecture chosen from searching across dimensions
    in the build_neural_classifier function

    :param rounds:
    :param prefix:
    :return:
    """
    depth = 3
    height = 16
    learning_rate = 0.005
    relu_fn = tf.nn.relu
    tf_apply_fn = partial(neural_base, depth, height, relu_fn)
    name = "%s%s-%s.%s.%s" % (prefix, depth, height, relu_fn.__name__, str(learning_rate))
    return NNClassifier(tf_apply_fn, name, rounds, learning_rate)


def run_classifier(classifier, train_matrix, train_labels, folds=5):
    print(" ======== ", classifier.__class__.__name__)
    if type(classifier) == SVMClassifier:
        X_train, X_test, y_train, y_test = train_test_split(train_matrix, train_labels,
                                                            test_size=0.30, random_state=42)
        classifier.fit(X_train, y_train)
        accuracy_train = classifier.score(X_train, y_train)
        accuracy_test = classifier.score(X_test, y_test)
        print(f"Classifier train accuracy:{accuracy_train}  and testaccuacy: {accuracy_test} on 30% test samples")
    elif type(classifier) == NBClassifier:
        r = classifier.get_cv_score(train_matrix, train_labels, 5)
        print("Score / {:<50} {:<25} {:<25} {:<25} ".format(*[str(classifier),
                                                              str(r["mean_train_score"]),
                                                              str(r["mean_test_score"]),
                                                              str(r["mean_test_score"] -
                                                                  r["mean_train_score"])]))
    else:
        # Convert to one-hot representation for DNN
        n_values = np.max(train_labels) + 1
        train_labels_reshape = np.eye(n_values)[train_labels]
        r = classifier.get_cv_score(train_matrix, train_labels_reshape, 2)
        # plt.clf()
        # plt.title("Loss, showing all updates".format(n_updates))
        # plt.plot(total_losses)
        print("Score / {:<50} {:<25} {:<25} {:<25} ".format(*[str(classifier),
                                                              str(r["mean_train_score"]),
                                                              str(r["mean_test_score"]),
                                                              str(r["mean_test_score"] - r["mean_train_score"])]))


@click.group()
def main():
    pass


@main.command('search')
@click.option('--data', default='./datasets/Jumble-for-JIRA.json', help='The path to the dataset.')
@click.option('--dir', 'directory', required=True, help='The path to the dataset.')
@click.option('--prefix', default='E', help='The dataset figure name')
def search(data, prefix, directory):
    json_files = [data]
    log.info(f'The directory value is: {directory}')
    if os.path.isdir(directory):
        json_files = [os.path.join(directory, f_json) for f_json in os.listdir(directory) if f_json.endswith('.json')]
        [log.debug(f'File found: {fname}') for fname in json_files]

    master_list = []
    for batch in json_files:
        with open(batch) as f:
            log.debug(f'Working with file batch: {batch}')
            dataset = json.load(f)
            master_list.extend(dataset)
        f.close()

    train_matrix, train_labels = vectorize_json(master_list, directory)
    # neural_net_classifiers = [SVMClassifier(), NBClassifier()]
    # neural_net_classifiers.extend(build_neural_classifier(2000, prefix))
    neural_net_classifiers = build_neural_classifier(2000, prefix)

    for classifier in neural_net_classifiers:
        print(" ======== ", classifier.__class__.__name__)
        if type(classifier) == SVMClassifier:
            X_train, X_test, y_train, y_test = train_test_split(train_matrix, train_labels,
                                                                test_size=0.30, random_state=42)
            classifier.fit(X_train, y_train)
            accuracy_train = classifier.score(X_train, y_train)
            accuracy_test = classifier.score(X_test, y_test)
            print(f"Classifier train accuracy:{accuracy_train}  and testaccuacy: {accuracy_test} on 30% test samples")
        elif type(classifier) == NBClassifier:
            r = classifier.get_cv_score(train_matrix, train_labels, 5)
            print("Score / {:<50} {:<25} {:<25} {:<25} ".format(*[str(classifier),
                                                                  str(r["mean_train_score"]),
                                                                  str(r["mean_test_score"]),
                                                                  str(r["mean_test_score"] -
                                                                      r["mean_train_score"])]))
        else:
            n_values = np.max(train_labels) + 1
            train_labels_reshape = np.eye(n_values)[train_labels]
            r = classifier.get_cv_score(train_matrix, train_labels_reshape, 2)
            # plt.clf()
            # plt.title("Loss, showing all updates".format(n_updates))
            # plt.plot(total_losses)
            print("Score / {:<50} {:<25} {:<25} {:<25} ".format(*[str(classifier),
                                                                  str(r["mean_train_score"]),
                                                                  str(r["mean_test_score"]),
                                                                  str(r["mean_test_score"] -
                                                                      r["mean_train_score"])]))


@main.command(name='production')
@click.option('--data', default='./datasets/Jumble-for-JIRA.json', help='The path to the dataset.')
@click.option('--dir', 'directory', required=True, help='The path to the dataset.')
@click.option('--prefix', default='P', help='The dataset figure name')
@click.option('--rounds', default=500, help='The number of rounds to backprop')
def production(data, directory, prefix, rounds):
    json_files = []
    log.info(f'The directory value is: {directory}')
    if os.path.isdir(directory):
        json_files = [os.path.join(directory, f_json) for f_json in os.listdir(directory) if f_json.endswith('.json')]
        [log.debug(f'File found: {fname}') for fname in json_files]

    master_list = []
    for batch in json_files:
        with open(batch) as f:
            log.debug(f'Working with file batch: {batch}')
            dataset = json.load(f)
            master_list.extend(dataset)
        f.close()

    train_matrix, train_labels = vectorize_json(master_list, directory)
    # neural_net_classifiers = [NBClassifier()] + build_neural_classifier(500)

    # SVM Linear Kernel Classifier
    svm_classifier = SVMClassifier()
    run_classifier(svm_classifier, train_matrix, train_labels)

    # Naieve Bayes Classifier
    nb_classifier = NBClassifier()
    run_classifier(nb_classifier, train_matrix, train_labels)

    # DNN Classifier
    dnn_classifier = production_neural_classifer(rounds, prefix)
    run_classifier(dnn_classifier, train_matrix, train_labels, folds=3)


if __name__ == "__main__":
    main()
