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
    p_value = 0.025

    set_names = ("mushrooms", "zoo")
    set_targets = (0, 17)
    # set_names = ("tiny_animal_set","tiny_animal_set")
    # set_targets = (2,2)
    for i in range( len(set_names) ):
        # Construct a new DataSet and corresponding DecisionTreeLearner
        dataset = DataSet(name=set_names[i], target=set_targets[i], attr_names=True)

        # UNPRUNED
        var = cross_validation(DecisionTreeLearner, dataset)
        errors = var[0]
        unpruned_err_mean = mean(errors)
        unpruned_err_std_dev = stdev(errors)
        unpruned_tree = var[1][0]
        unpruned_tree.chi_annotate(p_value)

        # Output unpruned tree
        format_output( (unpruned_err_mean, unpruned_err_std_dev, unpruned_tree) )

        # PRUNED
        var = cross_validation(DecisionTreeLearner, dataset)
        errors = var[0]
        pruned_err_mean = mean(errors)
        pruned_err_std_dev = stdev(errors)
        pruned_tree = var[1][0]
        pruned_tree.prune(p_value)
        pruned_tree.chi_annotate(p_value)

        # Output pruned tree
        format_output( (pruned_err_mean, pruned_err_std_dev, pruned_tree) )


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


def format_output(content):
    """
    Accepts a tuple with format (mean, std dev, tree)
    """
    if len(content) != 3 or not isinstance(content, tuple):
        print("Incorrect format provided to format_output function.")
        return
    try:
        mn = str(content[0])
        sd = str(content[1])
        tree = str(content[2])
    except Exception as e:
        print("Could not parse format_output() input as string.")
        return
    output = "Mean: " + mn + ", Standard Deviation: " + sd + ", Tree: \n\n" + tree + "\n"
    write_to_file(output)
    # print(output)


if __name__ == '__main__':
    main()
