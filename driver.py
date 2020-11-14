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
    data_set_names = "tiny_animal_set", 0
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

        format_output( (unpruned_err_mean, unpruned_err_std_dev, unpruned_tree) )
        return

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
    output = "Mean: " + mn + ", Standard Deviation: " + sd + ", Tree: \n\n" + tree
    # write_to_file(output)
    print(output)


if __name__ == '__main__':
    main()
