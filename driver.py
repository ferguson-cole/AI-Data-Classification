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
    


    
def main():
    """
    Machine learning with decision trees.
    Runs cross validation on data sets and reports results/trees
    """
    # construct_tree = DecisionTreeLearner

    # zoo_without_pruning = cross_validation(learner=DecisionTreeLearner, dataset=DataSet(name="zoo"))
    # print(zoo_without_pruning)
    # test_dataset = DataSet(name="mushrooms")
    # print(test_dataset.attr_names)

    # p_value
    dataset = DataSet(name="tiny_animal_set", target=2, exclude=[2])
    tree = DecisionTreeLearner(dataset)

    # TODO determine what to pass into this function then run it with 10 iterations for unpruned and 10 for pruned
    # def cross_validation(learner, dataset, *learner_posn_args, k=10, trials=1, **learner_kw_args):
    # cross_validation()
    # TODO figure out how to represent a pruned and un-pruned dataset
    p_value = 0.05
    tree.chi_annotate(p_value)

    # TODO get mean and std dev for each set of pruned and unpruned and write file writer
    init_file()
    write_to_file(tree) # Unpruned tree
    tree.prune(p_value)
    write_to_file(tree) # Pruned tree


def write_to_file(content, filename="output.txt"):
    f = open(filename, "a") # Create file if does not exist, will overwrite content
    # Write tree to file
    try:
        f.write(content)
    except TypeError as e:
        f.write(str(content))
    f.write("\n")
    f.close()


def init_file(filename="output.txt"):
    f = open(filename, "w") # If does not exist, create new; if exists, overwrite
    f.close()


if __name__ == '__main__':
    main()
