import math
import sys
from collections import namedtuple

import numpy as np
import scipy.stats
from scipy.stats import chi2

from ml_lib.ml_util import argmax_random_tie, normalize, remove_all, best_index, DataSet
from ml_lib.decision_tree_support import DecisionLeaf, DecisionFork
from ml_lib.utils import removeall, count


class DecisionTreeLearner:
    """DecisionTreeLearner - Class to learn decision trees and predict classes
    on novel examples
    """

    # Typedef for method chi2test result value (see chi2test for details)
    chi2_result = namedtuple("chi2_result", ('value', 'similar'))

    def __init__(self, dataset, debug=False, p_value=None):
        """
        DecisionTreeLearner(dataset)
        dataset is an instance of ml_lib.ml_util.DataSet.
        """

        # Hints: Be sure to read and understand the DataSet class
        # as you will use it throughout.

        # ---------------------------------------------------------------
        # Do not modify these lines, the unit tests will expect these fields
        # to be populated correctly.
        self.dataset = dataset

        # degrees of freedom for Chi^2 tests is number of categories minus 1
        self.dof = len(self.dataset.values[self.dataset.target]) - 1

        # Learn the decison tree
        self.tree = self.decision_tree_learning(dataset.examples, dataset.inputs)
        # -----------------------------------------------------------------

        self.debug = debug

        # # in class notes
        # # p_value is pruning
        # # sample code
        # mushrooms = DataSet(name="mushrooms")
        # mushrooms.examples[5]
        # mushrooms.attrs
        # mushroms.target
        #
        # mushrooms = DataSet(name="mushrooms", target=0)
        # mushrooms.examples
        # # use the information gain to figure out what question to ask

    def __str__(self):
        "str - Create a string representation of the tree"
        if self.tree is None:
            result = "untrained decision tree"
        else:
            result = str(self.tree)  # string representation of tree
        return result

    def decision_tree_learning(self, examples, attrs, parent=None, parent_examples=()):
        """
        decision_tree_learning(examples, attrs, parent_examples)
        Recursively learn a decision tree
        examples - Set of examples (see DataSet for format)
        attrs - List of attribute indices that are available for decisions
        parent - When called recursively, this is the parent of any node that
           we create.
        parent_examples - When not invoked as root, these are the examples
           of the prior level.
        """

        # Hints:  See pseudocode from class and leverage classes
        # DecisionFork and DecisionLeaf

        if len(examples) == 0:
            return DecisionLeaf(self.plurality_value(parent_examples), self.count_targets(parent_examples),
                                parent=parent)
        elif self.all_same_class(examples):
            # all the examples are of the same class
            # examples[0][self.dataset.target] represents a value of the classification goal
            return DecisionLeaf(examples[0][self.dataset.target],
                                self.count_targets(examples), parent=parent)
        # we no longer have questions
        elif len(attrs) == 0:
            return DecisionLeaf(self.plurality_value(examples), self.count_targets(examples),
                                parent=parent)
        else:
            # Enter the recursive algorithm
            a = self.choose_attribute(attrs, examples)
            # create a new tree rooted on most important question
            t = DecisionFork(a, self.count_targets(examples), self.dataset.attr_names[a],
                             parent=parent)
            # for each value v associated with attribute a:
            for (value, example) in self.split_by(a, examples):
                vexamples = self.decision_tree_learning(example, removeall(a, attrs), parent=t,
                                                        parent_examples=examples)
                t.add(value, vexamples)
            return t

    def plurality_value(self, examples):
        """
        Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality).
        """
        popular = argmax_random_tie(self.dataset.values[self.dataset.target],
                                    key=lambda v: self.count(self.dataset.target, v, examples))
        return popular

    def count(self, attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def count_targets(self, examples):
        """count_targets: Given a set of examples, count the number of examples
        belonging to each target.  Returns list of counts in the same order
        as the DataSet values associated with the target
        (self.dataset.values[self.dataset.target])
        """

        tidx = self.dataset.target  # index of target attribute
        target_values = self.dataset.values[tidx]  # Class labels across dataset

        # Count the examples associated with each target
        counts = [0 for i in target_values]
        for e in examples:
            target = e[tidx]
            position = target_values.index(target)
            counts[position] += 1

        return counts

    def all_same_class(self, examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][self.dataset.target]
        return all(e[self.dataset.target] == class0 for e in examples)

    def choose_attribute(self, attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs, lambda a: self.information_gain(a, examples))

    def information_gain(self, attr, examples):
        """Return the expected reduction in entropy for examples from splitting by attr."""
        # Represent the denominator
        total_examples = len(examples)

        # Compute the entropy
        entropy = self.information_content(self.count_targets(examples))
        # Compute the reminder where B is calculated from information content
        remainder = sum(np.multiply(float((np.divide(len(ex), total_examples))),
                                    self.information_content(self.count_targets(ex)))
                        # Returns the pairs
                        for (v, ex) in self.split_by(attr, examples))
        # Calculation from slides
        information_gain = entropy - remainder
        return information_gain

    def split_by(self, attr, examples):
        """split_by(attr, examples)
        Return a list of (val, examples) pairs for each val of attr.
        """
        return [(v, [e for e in examples if e[attr] == v]) for v in self.dataset.values[attr]]

    def predict(self, x):
        "predict - Determine the class, returns class index"
        return self.tree(x)  # Evaluate the tree on example x

    def __repr__(self):
        return repr(self.tree)

    @classmethod
    def information_content(cls, class_counts):
        """info = information_content(class_counts)
        Given an iterable of counts associated with classes
        compute the empirical entropy.
        Example: 3 class problem where we have 3 examples of class 0,
        2 examples of class 1, and 0 examples of class 2:
        information_content((3, 2, 0)) returns ~ .971
        Hint: Ignore zero counts; function normalize may be helpful
        """

        # Use the normalize method to get the results within a range of 1
        probability = normalize(removeall(0, class_counts))
        entropy = sum(-(p * np.log2(p)) for p in probability)
        return entropy
        # Hint: remember discrete values use log2 when computing probability

    def information_per_class(self, examples):
        """information_per_class(examples)
        Given a set of examples, use the target attribute of the dataset
        to determine the information associated with each target class
        Returns information content per class.
        """

        # Hint:  list of classes can be obtained from
        # self.data.set.values[self.dataset.target]

        # NOTES FROM OFFICE HOUR
        """
        10 things
        4 - one class / 6 - another class
        how much information is there for each of the classes (I = 1 / log_2(P))
        Use normalize func
        compute information for each one of these classes - do in the order in which they were expected
        dataset has attribute called values - tells you what the possible vares are (put in the same order)
        """

        # Compute the information for each one of these classes - do them in the order of which they are expected
        # Use a list to represent the info. per class in the order that is expected for target
        information_per_class = []
        probabilities = normalize(self.count_targets(examples))
        for p in probabilities:
            # calculate the probability of information (formula on slide 17)
            information_per_class.append(1 / np.log2(p))

        return information_per_class

    def prune(self, p_value):
        """Prune leaves of a tree when the hypothesis that the distribution
        in the leaves is not the same as in the parents as measured by
        a chi-squared test with a significance of the specified p-value.
        Pruning is only applied to the last DecisionFork in a tree.
        If that fork is merged (DecisionFork and child leaves (DecisionLeaf),
        the DecisionFork is replaced with a DecisionLeaf.  If a parent of
        and DecisionFork only contains DecisionLeaf children, after
        pruning, it is examined for pruning as well.
        """
        # Post order traversal to get near leaf nodes
        # Hint - Easiest to do with a recursive auxiliary function, that takes
        # a parent argument, but you are free to implement as you see fit.
        # e.g. self.prune_aux(p_value, self.tree, None)
        # I think the way we do this is pass in the completed decision tree then go to the nodes and reference the
        # chi2 values

        # Call the recursive helper function
        self.__prune_aux(self.tree, p_value)

    def __prune_aux(self, branch, p_value):
        if isinstance(branch, DecisionFork):
            # Check the children of the fork
            for children in branch.branches.values():
                # Check if the children are also decision forks
                if isinstance(children, DecisionFork):
                    # recursively go through the tree
                    children_branch = self.__prune_aux(children, p_value)
                    # Compute the chi2 stat
                    branch_test = self.chi2test(p_value, children_branch)
                    # Use the boolean value we are unsure about - from email, "examine the class distribution"
                    if branch_test.similar:
                        # If true, then replace the Decision Fork with a newly created Decision Leaf

                        # Start by creating a list that compares the whole tree from the values of the dict
                        post_order_value_list = [branch.branches.values()]
                        # Grab the index value of each child node
                        values = post_order_value_list.index(children_branch)

                        # Create another list to represent the keys of the dict
                        key_list = [branch.branches.keys()]
                        new_keys = key_list[values]

                        # "The node is then replaced with a new leaf node of that value."
                        # Result - Distribution - Parent
                        branch.branches[new_keys] = DecisionLeaf(self.dataset.values[self.dataset.target][best_index(
                            branch.distribution)], branch.distribution, branch.parent)
        return branch

    def chi_annotate(self, p_value):
        """chi_annotate(p_value)
        Annotate each DecisionFork with the tuple returned by chi2test
        in attribute chi2.  When present, these values will be printed along
        with the tree.  Calling this on an unpruned tree can significantly aid
        with developing pruning routines and verifying that the chi^2 statistic
        is being correctly computed.
        """
        # Call recursive helper function
        self.__chi_annotate_aux(self.tree, p_value)


    def __chi_annotate_aux(self, branch, p_value):
        """chi_annotate(branch, p_value)
        Add the chi squared value to a DecisionFork.  This is only used
        for debugging.  The decision tree helper functions will look for a
        chi2 attribute.  If there is one, they will display chi-squared
        test information when the tree is printed.
        """

        if isinstance(branch, DecisionLeaf):
            return  # base case
        else:
            # Compute chi^2 value of this branch
            branch.chi2 = self.chi2test(p_value, branch)
            # Check its children
            for child in branch.branches.values():
                self.__chi_annotate_aux(child, p_value)


    def chi2test(self, p_value, fork):
        """chi2test - Helper function for prune
        Given a DecisionFork and a p_value, determine if the children
        of the decision have significantly different distributions than
        the parent.
        Returns named tuple of type chi2result:
        chi2result.value - Chi^2 statistic
        chi2result.similar - True if the distribution in the children of the
           specified fork are similar to the the distribution before the
           question is asked.  False indicates that they are not similar and
           that there is a significant difference between the fork and its
           children
        """

        if not isinstance(fork, DecisionFork):
            raise ValueError("fork is not a DecisionFork")

        # Hint:  You need to extend the 2 case chi^2 test that we covered
        # in class to an n-case chi^2 test.  This part is straight forward.
        # Whereas in class we had positive and negative samples, now there
        # are more than two, but they are all handled similarly.

        # Don't forget, scipy has an inverse cdf for chi^2
        # scipy.stats.chi2.ppf

        # Setup delta
        delta = 0
        p_k_hat = 0
        children = fork.branches.values()

        # Handle parent calculations
        if fork.parent is None:
            p_dist = self.count_targets(self.dataset.examples)
        else:
            p_dist = fork.parent.distribution

        p_dist_len = len(p_dist)

        # Handle child calculations
        for child in children:
            c_dist = child.distribution

            # Calculation for variables needed to compute delta 
            for i in range(p_dist_len):
                p = p_dist[i]
                p_k = c_dist[i]
                n = 0
                n_k = 0
                for ii in range(len(p_dist)):
                    if ii == i:
                        continue
                    n += p_dist[ii]
                for ii in range(len(c_dist)):
                    if ii == i:
                        continue
                    n_k += c_dist[ii]

                p_k_hat = p * (p_k + n_k) / (p + n)

            if p_k_hat != 0:
                delta += ((p_k - p_k_hat)**2 / p_k_hat)
                print(delta)
            print("p_k -- " + str(p_k) + "  ||  p_k_hat -- " + str(p_k_hat) + "  ||  delta -- " + str(delta) + " || n -- " + str(n) + " || n_k -- " + str(n_k))
        print("done -- deltaresult -- " + str(delta))
        # Compute the probability density function
        ppf = chi2.ppf(1 - p_value, self.dof)
        # Handle output
        chi2result = namedtuple('chi2result', ['value', 'similar'])
        return chi2result(delta, (delta < ppf))

    def __str__(self):
        """str - String representation of the tree"""
        return str(self.tree)
