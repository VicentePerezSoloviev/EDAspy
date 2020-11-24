
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def __normalize__(array):
    """
    Normalization of the array
    :param array:
    :return: normalized array
    """

    if sum(array) == 0.0:
        return [0] * len(array)
    return [i / sum(array) for i in array]


class TransformationsFSEDA:

    generation = pd.DataFrame()
    output_plot = ''

    historic_best = []
    best_MAE = 99999999999
    best_ind = ''

    def __init__(self, max_it, dead_it, size_gen, alpha, vector, array_transformations, cost_function):
        self.max_it = max_it
        self.size_gen = size_gen
        self.alpha = alpha
        self.vector = vector
        self.dead_it = dead_it

        self.trunc_size = int(size_gen * alpha)

        self.array_transformations = ['basic'] + array_transformations
        self.variables = list(vector.columns)

        # check if cost_function is real
        if callable(cost_function):
            self.cost_function = cost_function
        else:
            raise Exception('ERROR setting cost function. The cost function must be a callable function')

    def __initialize_dirichlet__(self):
        """
        Initialization of the transformation database. Associate a key to each transformation.
        :return: updates the dictionary {transformation: key}
        """

        dirichlet = pd.DataFrame(columns=['transformation'] + self.variables)
        dirichlet['transformation'] = self.array_transformations
        dirichlet = dirichlet.set_index('transformation')

        for i in dirichlet.index:
            dirichlet.loc[i] = 1 / len(dirichlet)

        self.dirichlet = dirichlet

        # dictionary that specifies a keys to each transformation
        keys = {}
        for i in range(len(self.array_transformations)):
            keys[self.array_transformations[i]] = i + 2  # [2, inf]

        self.keys = keys

    def __new_individual__(self):
        """
        Creates a new individual.
        :return: dictionary {variable_name: number}
        """

        num_vars = len(self.variables)
        sample = list(np.random.uniform(low=0, high=1, size=num_vars))
        individual = {}
        index = 0

        for ind in self.variables:
            if float(self.vector[ind]) >= sample[index]:
                individual[ind] = 1
            else:
                individual[ind] = 0
            index = index + 1

        # if is one, then choose transformation or normal
        for ind in self.variables:
            if individual[ind] == 1:
                # assign the keys of the chosen transformation
                probabilities = list(self.dirichlet[ind].values)
                trans = list(self.dirichlet[ind].index)
                selection = np.random.choice(trans, 1, p=probabilities)
                individual[ind] = int(self.keys[selection[0]])

        return individual

    def new_generation(self):
        """
        Creates a new generation of individuals.
        :return: updates the generation DataFrame
        """

        gen = pd.DataFrame(columns=self.variables)

        while len(gen) < self.size_gen:
            individual = self.__new_individual__()
            gen = gen.append(individual, True)

            # drop duplicate individuals, to not calculate more than once
            gen = gen.drop_duplicates()
            gen = gen.reset_index()
            del gen['index']

        self.generation = gen

    def __getKeysByValue__(self, value_2_find):
        """
        Get a list of keys from dictionary which has the given value
        :param value_2_find: value to find in the dictionary
        :return: list of keys which match with value_2_find in the dictionary {transformation: key}
        """

        list_keys = list()
        list_items = self.keys.items()
        for item in list_items:
            if item[1] == value_2_find:
                list_keys.append(item[0])
        return list_keys

    def __check_individual__(self, individual):
        """
        Check the cost of the individual in the cost function.
        :param individual: dictionary of the respective individual.
        :return: cost of the individual ¿MAE?
        """

        variables = []  # list of variables included

        for i in self.variables:
            # if individual included in selection then != 0
            # else == 0
            if individual[i] != 0:
                # format: name + 'name_transformation'
                key = str(self.__getKeysByValue__(individual[i])[0])

                if key == 'basic':
                    variables.append(i)  # name
                else:
                    variables.append(i + key)  # name + name_transformation

        return self.cost_function(variables)

    # check the cost of each individual of the generation
    def check_generation(self):
        """
        Check the cost of each individual of the generation in the cost function.
        :return:
        """

        for ind in range(len(self.generation)):
            try:
                mae = self.__check_individual__(self.generation.loc[ind])
            except:
                raise Exception('ERROR: something went wrong calculating the cost of the individual: \n', str(ind))

            # print('ind: ', ind, ' MAE: ', mae)
            self.generation.loc[ind, 'MAE'] = mae

    # selection of the best individuals to mutate the next gen
    def individuals_selection(self):
        """
        Selection of the best individuals to mutate the next generation
        :return:
        """

        self.generation = self.generation.nsmallest(self.trunc_size, 'MAE')

    def update_vector_probabilities(self):
        """
        Re-build the vector of statistics based on the selection of the best individuals of the generation
        :return: update the vector of statistics
        """

        for ind in self.variables:
            # count how many 1s, 2s, 3s ...
            my_list = list(self.generation[ind].values)
            my_dict = {i: my_list.count(i) for i in my_list}

            # if not 0 in dictionary, then prob is 0
            if 0 not in my_dict:
                prob_vector = 0
            else:
                # if 0 in my_dict
                prob_vector = int(my_dict[0]) / len(self.generation)

            self.vector[ind] = 1 - prob_vector  # probability of being chosen

            for trans in self.dirichlet.index:
                key = int(self.keys[trans])
                # check if all values are counted
                if key not in my_dict:
                    prob_dirich = 0
                else:
                    prob_dirich = int(my_dict[key]) / len(self.generation)

                self.dirichlet.loc[trans, ind] = prob_dirich

        # normalize probabilities in dirichlet
        for ind in self.dirichlet.columns:
            values = list(self.dirichlet[ind].values)
            self.dirichlet[ind] = __normalize__(values)

    def __plot__(self):
        """
        Save a figure in the filename location with the EDA progress.
        output_plot must be overwritten previously.
        :return: save a fig in output_plot
        """

        if self.output_plot != '':
            iteration = list(range(len(self.historic_best)))
            plt.figure(figsize=(12, 8))
            plt.plot(iteration, self.historic_best)
            plt.title('EDA progression')
            plt.xlabel('iteration')
            plt.ylabel('MAE in model')
            plt.savefig(self.output_plot)

    def run(self, output=True):
        """
        Algorithm run execution.
        :param output: Boolean. If True then an output is printed in each iteration. Otherwise, not.
        :return: best_individual array, best MAE found double
        """

        convergence = 0
        self.__initialize_dirichlet__()

        for i in range(self.max_it):
            self.new_generation()
            self.check_generation()
            self.individuals_selection()
            self.update_vector_probabilities()

            best_mae_local = self.generation['MAE'].min()
            best_ind_local = []
            best = self.generation[self.generation['MAE'] == best_mae_local]
            best = best.reset_index()

            if len(best) > 1:
                best = best.loc[0]

            for var in self.variables:
                if int(best[var]) != 0:
                    # format: name + 'name_transformation'
                    string = var + str(self.__getKeysByValue__(int(best[var]))[0])
                    best_ind_local.append(string)

            self.historic_best.append(best_mae_local)  # save MAE

            '''if output:
                print(list(self.vector.loc[0]))
                print('Best of it.', best_mae_local)
                print(best_ind_local)'''

            # update best of model
            if self.best_MAE > best_mae_local:
                self.best_MAE = best_mae_local
                self.best_ind = best_ind_local

                # print('** Best MAE:', best_mae_local)
                convergence = 0
            else:
                convergence = convergence + 1

                if convergence == self.dead_it:
                    self.__plot__()  # save the fig of the progression
                    return self.best_ind, self.best_MAE

            if output:
                print('[iteration:', i, ']', best_mae_local)

        self.__plot__()  # save the fig of the progression
        return self.best_ind, self.best_MAE
