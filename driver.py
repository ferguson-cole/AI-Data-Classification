"""
Machine learning
decision trees
"""
import time

import scipy

from ml_lib.ml_util import DataSet

from decision_tree import  DecisionTreeLearner

from ml_lib.crossval import cross_validation

from statistics import mean, stdev


default_filename = "classifier.txt"

    
def main():
    """
    Machine learning with decision trees.
    Runs cross validation on data sets and reports results/trees
    """
    init_file()
    p_value = 0.05

    # data_set_names = ("mushrooms", "zoo")
    data_set_names = "tiny_animal_set"
    for set_name in data_set_names:
        
        dataset = DataSet(name=set_name, target=2, attr_names=True)
        tree = DecisionTreeLearner(dataset)

        tree.chi_annotate(p_value)

        # UNPRUNED
        var = cross_validation(DecisionTreeLearner, dataset)
        errors = var[0]
        unpruned_err_mean = mean(errors)
        unpruned_err_std_dev = stdev(errors)
        unpruned_tree = var[1][0]
        unpruned_output = "Mean: "
        print(unpruned_err_mean)
        return
        write_to_file(unpruned_err_mean)
        write_to_file(tree) # Write unpruned tree

        # PRUNED
        tree.prune(p_value)
        var = cross_validation(DecisionTreeLearner, dataset)
        errors = var[0]
        pruned_err_mean = mean(errors)
        pruned_err_std_dev = stdev(errors)
        pruned_tree = var[1][0]

        write_to_file(tree) # Pruned tree


def write_to_file(content, filename=default_filename):
    f = open(filename, "a") # Create file if does not exist, will overwrite content
    # Write tree to file
    try:
        f.write(content)
    except TypeError as e:
        f.write(str(content))
    f.write("\n")
    f.close()


def init_file(filename=default_filename):
    f = open(filename, "w") # If does not exist, create new; if exists, overwrite
    f.close()


if __name__ == '__main__':
    main()
