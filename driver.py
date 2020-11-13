"""
Machine 
decision trees
"""
import time
import numpy
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
    out = scipy.stats.chi2.cdf(2.3956,1)
    print(out)

if __name__ == '__main__':
    main()