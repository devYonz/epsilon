import click
import json
import numpy as np
import logging
import os, sys

from ml.feature import Feature
from ml.logistic import LogisticClassifier
from classifiers import build_neural_classifier
from processor import get_text_body, create_dictionary, transform_text

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logger = logging.getLogger(__name__)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(module)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[handler])

# log.setLevel(logging.DEBUG)

PROJECT_KEY = 'projects'
ISSUES_KEY = 'issues'
ASSIGNEE_KEY = 'assignee'


@click.command()
@click.option('--data', default='./datasets/Jumble-for-JIRA.json', help='The path to the dataset.')
@click.option('--spam', default=None, help='The path to the dataset.')
def main_yf(data, spam):
    print(f'Data set file recieved: {data}')
    with open(data) as f:
        dataset = json.load(f)

    issue_features = Feature()
    # TODO: capture the assignee class

    # Build the train matrix
    issues = []
    if PROJECT_KEY in dataset:
        for project in dataset.get(PROJECT_KEY):
            issues.extend(project.get(ISSUES_KEY))
    else:
        issues = dataset

    issues_lines = []
    assignees = []
    for issue in issues:
        words = get_text_body(issue)
        issues_lines.append(words)
        assignees.append(issue.get(ASSIGNEE_KEY))
    # Construct the words into vector of >5 occurance
    dictionary = create_dictionary(issues_lines)
    train_matrix = transform_text(issues_lines, dictionary)

    # Y label assignees -> int class number
    for class_name in list(set(assignees)):
        issue_features.set_target_id(class_name)
    assignee_vector = []
    for assignee in assignees:
        assignee_vector.append(issue_features.get_target_id(assignee))

    # Build the train feature vector
    issue_features.data = train_matrix
    issue_features.target = assignee_vector

    # classify and get output
    logistic = LogisticClassifier()
    logistic.fit(issue_features.data, issue_features.target)
    # test on a training set value
    x, y = issue_features.get_random_pairs(10)
    y_predict = logistic.predict(x)
    mse = 1/len(y) * np.linalg.norm(y_predict-y, ord=2)**2
    print(f'After training we recieved a MSE: {mse}')


@click.command()
@click.option('--data', default='./datasets/Jumble-for-JIRA.json', help='The path to the dataset.')
@click.option('--spam', default=None, help='The path to the dataset.')
def main(data, spam):
    print(f'Data set file recieved: {data}')
    with open(data) as f:
        dataset = json.load(f)

    issue_features = Feature()
    # TODO: capture the assignee class

    # Build the train matrix
    issues = []
    if PROJECT_KEY in dataset:
        for project in dataset.get(PROJECT_KEY):
            issues.extend(project.get(ISSUES_KEY))
    else:
        issues = dataset

    issues_lines = []
    assignees = []
    for issue in issues:
        words = get_text_body(issue)
        issues_lines.append(words)
        assignees.append(issue.get(ASSIGNEE_KEY))
    # Construct the words into vector of >5 occurance
    dictionary = create_dictionary(issues_lines)
    train_matrix = transform_text(issues_lines, dictionary)

    # Y label assignees -> int class number
    for class_name in list(set(assignees)):
        issue_features.set_target_id(class_name)
    assignee_vector = []
    for assignee in assignees:
        assignee_vector.append(issue_features.get_target_id(assignee))

    # Build the train feature vector
    issue_features.data = train_matrix
    issue_features.target = assignee_vector
    train_labels = issue_features.target
    n_values = np.max(train_labels) + 1
    train_labels_reshape = np.eye(n_values)[train_labels]

    neural_net_classifiers = build_neural_classifier(2000)

    for classifier in neural_net_classifiers:
        print(" ======== ", classifier.__class__.__name__)
        r = classifier.get_cv_score(train_matrix, train_labels_reshape, 10)
        # plt.clf()
        # plt.title("Loss, showing all updates".format(n_updates))
        # plt.plot(total_losses)
        print("Score / {:<50} {:<25} {:<25} {:<25} ".format(*[str(classifier),
                                                              str(r[
                                                                      "mean_train_score"]),
                                                              str(r[
                                                                      "mean_test_score"]),
                                                              str(r[
                                                                      "mean_test_score"] -
                                                                  r[
                                                                      "mean_train_score"])]))


if __name__ == "__main__":
    main()