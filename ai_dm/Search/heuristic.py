__author__ = 'sarah'


def get_heuristic(heuristic_name, problem):
    if 'zero' in heuristic_name:
        return zero_heuristic
    elif 'greedy' in heuristic_name:
        return greedy_heuristic
    elif 'multi_db' in heuristic_name:
        heur = db_multi_heuristic(problem)
        return heur.get_heur_val
    elif 'db' in heuristic_name:
        heur = db_heuristic(problem)
        return heur.get_heur_val


def zero_heuristic(node):
    return 0


def greedy_heuristic(node):
    if node.parent is None:
        return 0
    else:
        return 1


def goal_heuristic(node):
    if node.state.is_terminal:
        return 0
    else:
        return 1


class db_heuristic():

    def __init__(self, problem, key_indices=[0], split='true'):

        self.umd_problem = problem
        self.db = {}
        self.split = split
        self.key_indices = key_indices

    def get_key(self, design_node):

        key = []
        design_sequence = design_node.transition_path(False)
        params = []
        # iterate through modifications
        for modification in design_sequence:
            params = modification.get_params()
            # get the paramters of the modification
            index = 0
            for param in params:
                if index in self.key_indices:
                    param = param.replace(')', '')
                    if self.split:
                        key.append(param.split('-')[0])
                    else:
                        key.append(param)
                index += 1
        print('key is::::::: ')
        print(key)
        key = tuple(key)
        return [key, params]

    def get_heur_val(self, design_node):

        # get key
        [key, params] = self.get_key(design_node)
        # check if key exists in the db
        value = -1
        if key in self.db:
            value = self.db[key]

        else:
            # get the sequence of modifications - False flag means that the actual modifications are returned
            design_sequence = design_node.transition_path(False)
            padded_sequence = []
            padded_model = design_node.state

            for modification in design_sequence:
                # create a padded sequence, that includes only the valid modifications
                padded_modification = utils.get_padded_sequence(modification, design_node, self.key_indices)
                for modification in padded_modification:
                    padded_sequence.append(modification)
                    # True means the file name is random
                    padded_model = modification.apply(padded_model, True)

            # calculate the value of the padded model and keep the value in the db
            value = self.umd_problem.evaluate(padded_model)
            self.db[key] = value

        # return the heuristic value
        return value


class db_multi_heuristic():

    def __init__(self, umd_problem, key_indices_sets=[[0], [1], [2], ], split='true'):

        self.umd_problem = umd_problem
        self.db = {}
        self.split = split
        self.key_indices_sets = key_indices_sets

    def get_key(self, design_node, key_indices):

        key = []
        design_sequence = design_node.transition_path(False)
        params = []
        # iterate through modifications
        for modification in design_sequence:
            params = modification.get_params()
            # get the paramters of the modification
            index = 0
            for param in params:
                if index in key_indices:
                    param = param.replace(')', '')
                    if self.split:
                        key.append(param.split('-')[0])
                    else:
                        key.append(param)
                index += 1
        print('key is::::::: ')
        print(key)
        key = tuple(key)
        return [key, params]

    def get_heur_val(self, design_node):

        heur_values = []

        for key_indices in self.key_indices_sets:

            # get key
            [key, params] = self.get_key(design_node, key_indices)

            # check if key exists in the db
            value = -1
            if key in self.db:
                value = self.db[key]

            else:
                # get the sequence of modifications - False flag means that the actual modifications are returned
                design_sequence = design_node.transition_path(False)
                padded_sequence = []
                padded_model = design_node.state

                for modification in design_sequence:
                    # create a padded sequence, that includes only the valid modifications
                    padded_modification = utils.get_padded_sequence(modification, design_node, key_indices)
                    for modification in padded_modification:
                        padded_sequence.append(modification)
                        # True means the file name is random
                        padded_model = modification.apply(padded_model, True)

                # calculate the value of the padded model and keep the value in the db
                value = self.umd_problem.evaluate(padded_model)
                self.db[key] = value

            heur_values.append(value)

        min_val = min(heur_values)

        # return the heuristic value
        return min_val