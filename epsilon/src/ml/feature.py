import random


class Feature(object):
    """
    The class for feature vectors that are used to implement classifications

    """
    def __init__(self):
        #: X - 2D feature vector
        self.data = []
        #: Y - 1D output vector of classes
        self.target = []
        #: feature names - Name of the feature column
        self.feature_names = []
        #: target names - Name of the corresponding class id
        self.target_names = []
        #: target class - The class id of the corresponding name
        self.target_ids = {}

    # def add_sample(self, x, y):
    #     """
    #
    #     :param x:
    #     :param y:
    #     :return:
    #     """
    #     if self.get_target_id(y):
    #         self.data.append(x)
    #     else:
    #         self.set_target_id(y)
    #     self.target.append(self.get_target_id(y))

    def set_feature_names(self, names: list)-> None:
        """ Set the feature column names for the data

        :param names:
        :return:
        """
        self.feature_names = names

    def set_target_names(self, targets: list) -> None:
        """ Set the label name for the data

        :param targets:
        :return:
        """
        self.target_names = targets
        for target, class_id in enumerate(targets):
            self.target_ids[target] = class_id

    def set_target_id(self, target: str) -> None:
        """ Set the label name for the data

        :param target: a string value to add
        :return:
        """
        self.target_names.append(target)
        self.target_ids[target] = len(self.target_names) - 1 if len(self.target_names) else 0

    def get_feature_name(self, id: int)-> str:
        """ Get the string representation of a class

        :param id:
        :return:
        """
        return self.feature_names[id]

    def get_target_name(self, id : int)-> str:
        """ Transform the class label into class name

        :param id:
        :return:
        """
        return self.target_names[id]

    def get_target_id(self, name: str)-> int:
        """ Transform the string class name to id
        :param str name: the name of the class
        :return:
        """
        return self.target_ids.get(name)

    def get_target_names(self, ids : list)-> list:
        """ Transform the array of labels into string representation

        :param ids:
        :return:
        """
        s_labels = []
        for id in ids:
            s_labels.append(self.target_names[id])

    def get_random_pairs(self, count):
        sample_size = len(self.data)
        x = []
        y = []
        for i in range(count):
            idx = random.randint(0, sample_size)
            x.append(self.data[idx])
            y.append(self.target[idx])
        return x,y

    def flush(self):
        """ Print out the data with headers and transformed

        :return:
        """
        raise NotImplementedError