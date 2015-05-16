
mining-quest
===================

Naive Bayes implementation on yelp.com data
-------------------
Instructions for BoW_NB.py:
- Code will run for around 42 seconds with one 5000 learning dataset and one 5000 input
dataset.

- The code will also throw an input error if anything other than a 5 or a 7 is passed
for a classification task. The 5 or 7 corresponds to the particular feature in the 
yelp.com dataset. (5 = Funny, 7 = Stars)

Expected Input:
- trainingDataFilename 	= sys.argv[1]
- testDataFilename 		  = sys.argv[2]
- classLabelIndex 		  = sys.argv[3]
- printTopWords 		  = int(sys.argv[4])

Example
- python BoW_NB.py stars_data.csv stars_data.csv 7 1
