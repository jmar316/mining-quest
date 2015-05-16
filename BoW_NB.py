import csv, string, sys, re, random, itertools
from collections import Counter
from collections import defaultdict

# Jason Martins - contact: jason.j.martins@gmail.com
# Bag-of-words Naive Bayes Implementation

#####################################
def csv2LISTDICT(file_arg):
	#	Local Variables
	table_list = []						# our result of a list of dictionaries
	temp_row = {}  						# hold our temporary rows
	
	#	Begin
	file = open(file_arg)
	data = csv.reader(file)
	
	for row in data:
		if data.line_num == 1: 			# Extract header line to be used in zip later
			file_header = row
		else:
			temp_row = dict(zip(file_header, row))
			
			# Remove punctuation (incl. \n) &  mark lower case
			temp_row["text"] = string.replace(temp_row["text"],'.',' ')
			temp_row["text"] = temp_row["text"].translate(None, string.punctuation)
			temp_row["text"] = temp_row["text"].lower()
			temp_row["text"] = string.replace(temp_row["text"],'\n',' ')
			
			table_list.append(temp_row)
	
	return table_list
#####################################

def freqCOUNT(table,printTopWords):

	#	Local Variables
	bag_of_words = ''
	
	#	Begin - create one giant string
	for i in range(len(table)):
		bag_of_words = ''.join([table[i]["text"],bag_of_words])
	
	# Create our big bag of words (only grab 201-2200)
	counter_result = Counter(re.findall(r'\w+',bag_of_words)).most_common()[201:2200]
	
	freq_dict = defaultdict(list)
	for key, value in counter_result:
		freq_dict[key].append(value)
	
	if printTopWords == 1:
		for i in range(1,11):
			print('WORD{0} {1}').format(i, sorted(freq_dict, key=freq_dict.get, reverse=True)[i-1])
	
	return freq_dict
	
#####################################
				
def assembleBINARY(train_table,freq_dict):
	#	Local Variable
	binFeature_dict  = defaultdict(list)
	
	#	Begin - create a dictionary with 2000 keys -> top words
	#   	each key will map to a list to hold the binary features
	for key in freq_dict:
		binFeature_dict[key] = [0] * len(train_table)
		
	# Now search across the training table and assemble features
	for i in range(len(train_table)):
		review_vector = grepString(train_table[i]["text"])
		for k in binFeature_dict.keys():
			#if grepString(k,train_table[i]["text"]):
			if k in review_vector:
				binFeature_dict[k][i] = 1
				
	
	#This will hold key -> word and a value 
	# -> list of whether the word appeared or not in a given review
	return binFeature_dict
	
#####################################	
def grepString(text):
	# Takes a review and separates all words using regex
	
	review_vector = re.findall(r'\w+',text,flags=re.I)
	
	return review_vector

#####################################
def assembleBINARY_classTask(train_table,feature):
	#	Local Variable
	binFeature_dict  = defaultdict(list)
	
	#	Begin - create a dictionary with 1 key -> isFunny or isPositive
	#   	each key will map to a list to hold the binary features
	if feature == 5:
		binFeature_dict['isFunny'] = [0] * len(train_table)
		
		# Now search across the training table and assemble features
		for i in range(len(train_table)):
			#for k in binFeature_dict.keys():
			if int(train_table[i]['funny']) > 0:
				binFeature_dict['isFunny'][i] = 1
	else:
		binFeature_dict['isPositive'] = [0] * len(train_table)	
		
		# Now search across the training table and assemble features
		for i in range(len(train_table)):
			#for k in binFeature_dict.keys():
			if int(train_table[i]['stars']) == 5:
				binFeature_dict['isPositive'][i] = 1			
	
	return binFeature_dict

#####################################
def calcNBC(testData,learn_classTaskFeature,learn_bag0wordsFeature,test_classTaskFeature,test_bag0wordsFeature):
	
	#	Local Variable(s)
	classTask_word = defaultdict(dict) 	# - creating a dictionary of dictionaries 
										# - to hold p(word1|isFunny), p(word2|isFunny)
	
	p_classTask  = defaultdict(dict)   	# - hold p(isFunny) values	
	
	prediction_classTask = defaultdict(list) # to hold our predicted values
	
	classTask = ''
	yes_total = no_total = 0
	incorrect_classifications = float(0)
	
	
	# Begin
	###########################################
	# Prepare p(isFunny or isPositive) counts #
	###########################################
	
	if 'isFunny' in learn_classTaskFeature:	# will be faster and use dict hashing & no linear
		classTask = 'isFunny'
	else:
		classTask = 'isPositive'
	
	prediction_classTask[classTask] = [0] * len(testData)	
	p_classTask[classTask]['yes'] = float(0)
	p_classTask[classTask]['no'] = float(0)
	
	for i in range(len(learn_classTaskFeature[classTask])):
		if learn_classTaskFeature[classTask][i] == 1:
			p_classTask[classTask]['yes'] +=1
			yes_total += 1
		else:
			p_classTask[classTask]['no'] +=1
			
	no_total = len(learn_classTaskFeature[classTask]) - yes_total
	
	p_classTask[classTask]['yes'] /= len(learn_classTaskFeature[classTask])
	p_classTask[classTask]['no']  /= len(learn_classTaskFeature[classTask])
	
	p_classTask[classTask]['countYES'] = yes_total
	p_classTask[classTask]['countNO'] = no_total
	
	
	###########################################
	# p(word1|isFunny)...p(word2|isFunny).... #
	###########################################
	
	# Reference
	# yes_yes = isFunny = 1 & word = 1  
	# yes_no  = isFunny = 1 & word = 0
	# no_yes = isFunny = 0 & word = 1
	# no_no = isFunny = 0 & word = 0 
	
	for word_key in learn_bag0wordsFeature.iterkeys():
		classTask_word[word_key]['yes_yes'] = float(0)
		classTask_word[word_key]['yes_no'] = float(0)
		classTask_word[word_key]['no_yes'] = float(0)
		classTask_word[word_key]['no_no'] = float(0)
		
		for i in range(len(learn_classTaskFeature[classTask])):
			if learn_classTaskFeature[classTask][i] == 1:
				if learn_bag0wordsFeature[word_key][i] == 1:
					classTask_word[word_key]['yes_yes'] += 1
				else:
					classTask_word[word_key]['yes_no'] += 1
			else:
				if learn_bag0wordsFeature[word_key][i] == 1:
					classTask_word[word_key]['no_yes'] += 1
				else:
					classTask_word[word_key]['no_no'] += 1
		
		#Laplace Correction
		
		#Numerator
		classTask_word[word_key]['yes_yes'] += 1
		classTask_word[word_key]['yes_no'] += 1
		classTask_word[word_key]['no_yes'] += 1
		classTask_word[word_key]['no_no'] += 1
		
		#Denominator
		classTask_word[word_key]['yes_yes'] /= (p_classTask[classTask]['countYES']+2)			
		classTask_word[word_key]['yes_no'] /= (p_classTask[classTask]['countYES']+2)		
		classTask_word[word_key]['no_yes'] /= (p_classTask[classTask]['countNO']+2)
		classTask_word[word_key]['no_no'] /= (p_classTask[classTask]['countNO']+2)
	
	for i in range(len(testData)):
		rolling_probClassTRUE = rolling_probClassFALSE = float(1)
		
		for key in test_bag0wordsFeature.iterkeys():
			if test_bag0wordsFeature[key][i] == 1:
				rolling_probClassTRUE *= classTask_word[key]['yes_yes']
				rolling_probClassFALSE *= classTask_word[key]['no_yes']
			else:
				rolling_probClassTRUE *= classTask_word[key]['yes_no']
				rolling_probClassFALSE *= classTask_word[key]['no_no']
		
		rolling_probClassTRUE *= p_classTask[classTask]['yes']
		rolling_probClassFALSE *= p_classTask[classTask]['no']

		if rolling_probClassTRUE >= rolling_probClassFALSE:
			prediction_classTask[classTask][i] = 1
		else:	
			prediction_classTask[classTask][i] = 0

	# Checking how many incorrect classifications there are
	for i in range(len(test_classTaskFeature[classTask])):
		if prediction_classTask[classTask][i] != test_classTaskFeature[classTask][i]:
			incorrect_classifications += 1.0 
	
	# Calculating Zero-One Loss (incorrect/ total trials)
	zero_one_loss = incorrect_classifications/float(len(test_classTaskFeature[classTask]))
	print 'ZERO-ONE-LOSS {0}'.format(round(zero_one_loss,4))	
	
	return p_classTask

######## Main ###########################################################################

# Capture arguments from command line
trainingDataFilename = sys.argv[1]	# Training Set Data
testDataFilename = sys.argv[2]		# Test Set Data
classLabelIndex = sys.argv[3]		# Classification feature... 5 = Funny, 7 = Stars
printTopWords = int(sys.argv[4])	# If set to 1 will print out the top 10 unique words

train_list 	= []
test_list 	= []
train_test_split = float(0.50)
# Training Set Processing (going to print out top words for only training set)
# Result: A dict with the bag-of-words features

train_table_list = csv2LISTDICT(trainingDataFilename)

# Randomly sampling the same data file
# 90% for training, 10% for testing 
# this will see how our training set size affects our result

for j in range(int(len(train_table_list)*train_test_split)):
	random_training_dict = random.choice(train_table_list)
	while random_training_dict in train_list:
		random_training_dict = random.choice(train_table_list)
	train_list.append(random_training_dict)

test_list = [item for item in train_table_list if item not in train_list]	

freq_dict = freqCOUNT(train_list,printTopWords) 
training_bagOwords_feature = assembleBINARY(train_list,freq_dict)

# Test Set Processing
# Result: A dict with the bag-of-words features

#Since we are sampling, no need to look at the 2nd file name passed as our test
#test_table_list = csv2LISTDICT(testDataFilename)

test_bagOwords_feature = assembleBINARY(test_list,freq_dict)

# Classification Tasks on both the training and test sets will depend on the 
# 3rd parameter passed to the main python script (only took 5 & 7 into consideration)
# Either classify the Star Rating (7) or the Funny Rating (5)
# calcNBC will calculate our Naive Bayes
if int(sys.argv[3]) == 5:
	training_classification_feature = assembleBINARY_classTask(train_list,5)
	test_classification_feature = assembleBINARY_classTask(test_list,5)
	calcNBC(test_list,training_classification_feature,training_bagOwords_feature,test_classification_feature,test_bagOwords_feature)
elif int(sys.argv[3]) == 7:
	training_classification_feature = assembleBINARY_classTask(train_list,7)
	test_classification_feature = assembleBINARY_classTask(test_list,7)
	calcNBC(test_list,training_classification_feature,training_bagOwords_feature,test_classification_feature,test_bagOwords_feature)
else:
	print 'ERROR: Invalid Classification Task #...program will not proceed!'

