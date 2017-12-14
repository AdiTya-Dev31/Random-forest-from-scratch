Disclaimer: The accuracy values vary a lot because of random datasets. Please run the code at least twice to get a better picture. 

The approach taken is to minimize the running time and obtain higher accuracy.

Hence input parameters, no_of forest is just 6

Bootstrap : 
Pick a random number p
Dataset size is about 2/3 of total dataset length.
Starting from index p, choose data of size dataset from the array passed.

Attribute selection: Compute entropy based on median split and information gain for the sorted tuple (attribute value, target) for each attribute

This will give us a tuple of format (attribute_1, Info_gain_1)

Now sort the attributes based on information gain.

Pick top 3 attributes to construct the tree

When to stop splitting:

Stop splitting when 

	1) all elements of subtree belong to the same class 
	2) number of elements of each subtree is same as of main tree (Splitting does not happen here because the current split does not add value. In this case the majority of the class in the split is taken into consideration. Accomplished using bincount and argmax of numpy)
	3) there are no more elements in the subtree
	
Output Obtained:
OOB Error Estimate:  3.26485892699
accuracy: 0.9560