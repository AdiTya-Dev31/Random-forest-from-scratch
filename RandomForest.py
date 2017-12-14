import csv
import numpy as np  # http://www.numpy.org
import ast
from random import randint
from math import ceil,log
from operator import itemgetter
from collections import Counter
 
 
"""
Here, X is assumed to be a matrix with n rows and d columns
where n is the number of total records
and d is the number of features of each record
Also, y is assumed to be a vector of d labels
 
XX is similar to X, except that XX also contains the data label for
each record.
"""
 
"""
This skeleton is provided to help you implement the assignment. It requires
implementing more that just the empty methods listed below.
 
So, feel free to add functionalities when needed, but you must keep
the skeleton as it is.
"""
 
class RandomForest(object):
    class __DecisionTree(object):
        def __init__(self): self.tree = {}
        score_list = []                             # corresponding to records in the bootstrapping dataset 
        attribute_score = []
        attribute_to_split = []

        """This method calculated the entropy(impurity measure for each individual attribute. )"""
        
        def calc_entropy(self, attribute_tuple_list):
            no_of_entries = len(attribute_tuple_list)
            attribute_value = []
            target_class = []
            for entry in attribute_tuple_list:
                p,q = entry # here p is attribute value and q is the target
                attribute_value.append(p)
                target_class.append(q)
            counter_entropy = Counter(target_class)
            count_zero = float(counter_entropy[0.0]) / no_of_entries
            count_one = float(counter_entropy[1.0]) / no_of_entries
            if count_zero == 0:
                entropy_score = -1 * (count_one * log(count_one,2))
            elif count_one == 0:
                entropy_score = -1 * (count_zero * log(count_zero,2))
            else:
                entropy_score = -1 * (count_zero * log(count_zero,2)) + -1 * (count_one * log(count_one,2))
            return entropy_score

        def decide_attributes(self, attribute_tuple_list):
            if len(attribute_tuple_list) < 300: # Repeat splitting until the number of entries of subtree reduces to less than 300
                return
            self.score_list.append(self.calc_entropy(attribute_tuple_list))
            mid_value = int(len(attribute_tuple_list)/2)
            self.decide_attributes(attribute_tuple_list[:mid_value])
            self.decide_attributes(attribute_tuple_list[mid_value+1:])
 
        def build_tree(self, X, y):
            length_y = len(y)
            if not length_y:
                return 0
            total = np.sum(y)
            if not total or total==length_y:
                return y[0]
            expected = length_y/2
            perfect_split = float('inf')
            medians = np.median(X, axis=0)
            l_X = []
            selection = []
            column_no = -1
            for i in xrange(3):
                '''Split based on the attribute with maximum information gain. Split only when the variation is less than that of its parent'''
                att = self.attribute_to_split[i]
                median = medians[att]
                selected = np.where(X[:,att]<=median)
                sel_X = X[selected]
                current_split = abs(expected-len(sel_X))
                if current_split<perfect_split:
                    perfect_split = current_split
                    l_X = sel_X
                    selection = selected
                    column_no = att
            if not len(l_X) or len(l_X) == len(X):
                return np.argmax(np.bincount(y.astype(int)))
            left = self.build_tree(l_X, y[selection])
            selection_right = np.where(X[:,column_no]>medians[column_no])
            right = self.build_tree(X[selection_right], y[selection_right])
            return {'attribute': column_no, 'split_value': np.max(l_X[:,column_no]), 'left': left, 'right':right}
 
        def learn(self, X, y):
            # TODO: train decision tree and store it in self.tree
            self.attribute_score = []
            transposed_XX = X.T
            for index,attribute in enumerate(transposed_XX):
                information_gain = []
                if index != len(transposed_XX) - 1:
                    attribute_list = zip(X[:,index], np.array(y))  # Returns a tuple (attribute_value, y) for all the entries in X
                    sorted_attribute = sorted(attribute_list, key = itemgetter(0), reverse=True) # Sort tuple array based on attribute value
                    self.score_list = []
                    self.decide_attributes(sorted_attribute)
                    for idx,entropy in enumerate(self.score_list):
                        if idx < len(self.score_list) - 2:
                            information_gain.append(self.score_list[idx+1]- self.score_list[idx])
                    self.attribute_score.append((index, max(information_gain)))
            self.attribute_score = sorted(self.attribute_score, key = itemgetter(1), reverse=True)
            self.attribute_to_split = []
            for i in xrange(3):
                self.attribute_to_split.append(self.attribute_score[i][0])
            self.tree = self.build_tree(X, y)
 
        def classify(self, record):
            # TODO: return predicted label for a single record using self.tree
            tree = self.tree
            while type(tree)==dict:
                att = tree['attribute']
                if record[att]<=tree['split_value']: # Same logic as binary search
                    tree = tree['left']
                else:
                    tree = tree['right']
            return tree
 
    num_trees = 0
    decision_trees = []
    bootstraps_datasets = [] # the bootstrapping dataset for trees
    bootstraps_labels = []   # the true class labels,
                             # corresponding to records in the bootstrapping dataset
 
    def __init__(self, num_trees):
        # TODO: do initialization here.
        self.num_trees = num_trees
        self.decision_trees = [self.__DecisionTree() for i in range(num_trees)]
 
    def _bootstrapping(self, XX, n):
        # TODO: create a sample dataset with replacement of size n
        #
        # Note that you will also need to record the corresponding
        #           class labels for the sampled records for training purpose.
        #
        # Referece: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        p = randint(1,n) #Randomly choose a number p less than n 
        data_set_size = int(0.66 * n)
        if n - p >= data_set_size:  # here the dataset size is constant though the data is picked from different portions of XX based on the value of p
            return (XX[p:p+data_set_size,:-1], XX[:,-1])
        else:
            one = XX[p:, :-1].tolist()
            two = XX[:data_set_size-p, :-1].tolist()
            one.extend(two) # append p to end of XX and beginning of XX to remaining dataset
            x = np.array(one, dtype = float)
            one_y = XX[p:, -1].tolist()
            two_y = XX[:data_set_size-p, -1].tolist()
            one_y.extend(two_y)
            y = np.array(one_y, dtype=float)
            return (x,y)
        
 
    def bootstrapping(self, XX):
        # TODO: initialize the bootstrap datasets for each tree.
        XX = np.array(XX, dtype = float)
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)
 
    def fitting(self):
        # TODO: train `num_trees` decision trees using the bootstraps datasets and labels
        for i in xrange(self.num_trees):
            self.decision_trees[i].learn(self.bootstraps_datasets[i], self.bootstraps_labels[i])
 
    def voting(self, X):
        y = np.array([], dtype = int)
        no_of_entries = 0
        sum_OOB = 0
        for record in X:
            # TODO: find the sets of proper trees that consider the record
            #       as an out-of-bag sample, and predict the label(class) for the record.
            #       The majority vote serves as the final label for this record.
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                if record in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)                   
            counts = np.bincount(votes)
            no_of_entries += 1
            total = np.argmin(counts) + np.argmax(counts)
            if total == 0:
                total = counts[0]
                OOB_error = 0
            else:
                OOB_error = float(np.argmin(counts))/total
            
            sum_OOB += (OOB_error/total)/no_of_entries
            # returns a dictionary similar to Counter of python
            if len(counts) == 0:
                # TODO: special handling may be needed.
                y = np.append(y, 0)
            else:
                y = np.append(y, np.argmax(counts))  # return element with maximum count from the counts dicitonary
        print 'OOB Error Estimate: ', sum_OOB 
        return y
 
def main():
    X = list()
    y = list()
    XX = list() # Contains data features and data labels
 
    # Note: you must NOT change the general steps taken in this main() function.
 
    # Load data set
    with open("data.csv") as f:
        next(f, None)
 
        for line in csv.reader(f, delimiter = ","):
            X.append(line[:-1])
            y.append(line[-1])
            xline = [ast.literal_eval(i) for i in line]
            XX.append(xline[:])
 
    # Initialize according to your implementation
    forest_size = 6
 
    # Initialize a random forest
    randomForest = RandomForest(forest_size)
 
    # Create the bootstrapping datasets
    randomForest.bootstrapping(XX)
 
    # Build trees in the forest
    randomForest.fitting()
 
    # Provide an unbiased error estimation of the random forest
    # based on out-of-bag (OOB) error estimate.
    # Note that you may need to handle the special case in
    #       which every single record in X has used for training by some
    #       of the trees in the forest.
    y_truth = np.array(y, dtype = int)
    X = np.array(X, dtype = float)
    y_predicted = randomForest.voting(X)
    #print np.sum(y_truth),np.sum(y_predicted)
    #results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]
    results = [prediction == truth for prediction, truth in zip(y_predicted, y_truth)]
 
    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))
    print "accuracy: %.4f" % accuracy
 
main()
