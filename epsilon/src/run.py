import click
import json
import numpy as np


from ml.feature import Feature
from ml.logistic import LogisticClassifier
from processor import get_text_body, create_dictionary, transform_text

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


PROJECT_KEY = 'projects'
ISSUES_KEY = 'issues'
ASSIGNEE_KEY = 'assignee'


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

    # classify and get output
    logistic = LogisticClassifier()
    logistic.fit(issue_features.data, issue_features.target)
    # test on a training set value
    x, y = issue_features.get_random_pairs(10)
    y_predict = logistic.predict(x)
    mse = 1/len(y) * np.linalg.norm(y_predict-y, ord=2)**2
    print(f'After training we recieved a MSE: {mse}')


if __name__ == "__main__":
    main()