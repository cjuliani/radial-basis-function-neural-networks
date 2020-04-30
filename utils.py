import csv,io,ast
import numpy as np

def read_csv(filename):
	data_tabs = []
	with io.open(filename, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')

		# Skip header
		header = next(reader)
		
		for row in reader:
			# Convert row values from string to their own type
			row_TMP = [ast.literal_eval(i) for i in row]
			data_tabs.append(row_TMP)
	
	return np.asarray(data_tabs)

def convert_labels(csvData):
	'''
	Convert labels into ternary form {0,1,2}
	Notes: 
		there are 5 classes/deposit types (0 to 4)
		deposit target corresponds to 1, other deposits equal 2, and no deposit is 0
	'''
	csvCopy = csvData[:,1:].copy()	# starts from column 1 to rule out OID (identification number on ArcGIS)
	lb_n = []
	for i in range(len(csvCopy[:,-3])):
		if csvCopy[i,-3] == 4.0:
			lb_n.append(1)
		elif (csvCopy[i,-3] != 4.0) and (csvCopy[i,-3] != 0):
			lb_n.append(2)
		else:
			lb_n.append(0)

	# Replace labels in the copied table
	lb_n = np.asarray(lb_n)
	csvCopy[:,-3] = lb_n
	return csvCopy

def convert_rock_units(runits):
	'''
	Transform rock units ("geology") to {0,1} labels
	Notes:
		Rock unit = 1 (favorable for mineralization) | Identified in Finnmark by 55,28,35 (greenstone and volcanic rocks)
	 	Others = 0 (non favorable)
	'''
	rck_n = np.asarray( [1 if (runits[i,0] == 55) or (runits[i,0] == 28) or (runits[i,0] == 35) else 0 for i in range(runits.shape[0])] )
	runits[:,0] = rck_n		# number of cells studied = 108805
	return runits

def clusterDep(data,idxs,dist):
	'''
	(1) Clusters deposits given their distance/coordinates that are close to avoid preferential sampling
	and (2) re-define the deposit patterns by taking the average of anomalies from deposits close to each others
	:param data: data from grid cells
	:param idxs: indexes of depist patterns
	:param dist: maximum distance for clustering
	'''
	d_similar = []
	for i in range(len(idxs)):
		# retrieves indexes of patterns not target deposits
		ls = [e for e in idxs if e != idxs[i]]

		# retrieves coordinates (x,y) of target deposits to calculate
		# distances between them
		d_TMP = []
		ls_ = [e for sub in d_similar for e in sub]
		if idxs[i] not in ls_:
			for j in ls:
				x1 = data[idxs[i]][-2]
				x2 = data[j][-2]
				y1 = data[idxs[i]][-3]
				y2 = data[j][-3]
				distance = np.sqrt( np.square(x1 - x2) + np.square(y1 - y2) )

				if distance <= dist:
					#
					if j not in ls_:
						d_TMP.append(j)
			d_TMP.append(idxs[i])
			d_similar.append(d_TMP)
		else:
			pass
	# Average pattern data given the cluster elements
	# e.g. if cluster of 3 deposits, then calculate average magnetic anomaly
	dp_ = clusterMean(data=data,clusters=d_similar)
	return dp_

def clusterMean(data,clusters):
	'''
	:param data: data from grid cells
	:param clusters: clusters of pattterns in consideration
	'''
	d_patterns = []
	for i in clusters:
		TMP_ = []
		# takes the geology of first deposit from cluster
		TMP_.append( data[i[0],0] )
		for j in range(1,len(data[0,:-3])):
			# average data for cluster i, which can be of any length (1,2,3...)
			TMP_.append( np.mean( data[i,j] ) )
		TMP_.append( data[i[0],-3] )
		TMP_.append( data[i[0],-2] )
		TMP_.append( data[i[0],-1] )
		d_patterns.append(TMP_)
	return np.asarray(d_patterns)

def get_fingerprints(csvData,cluster_dist=250):
	'''
	(1) Collect deposit "fingerprints" (or patterns) from favorable rock units
	i.e. geoscience observations (lithology, magnetic anomaly, geochemistry...) for a given deposit
	(2) and cluster fingerprints that are close to each others

	:param cluster_dist: max distance to cluster deposits
	'''
	deposit_patterns, nondeposit_patterns = [],[]
	deposit_patterns_idxs, nondeposit_patterns_idxs = [],[]
	idx = 0
	for i in csvData:
		if i[0] == 1 and i[-3] == 1:
			# if favorable geology and target deposit are True...
			deposit_patterns.append(list(i))
			deposit_patterns_idxs.append(idx)
		elif i[-3] == 2:
			# if non-target deposit
			nondeposit_patterns.append(list(i))
			nondeposit_patterns_idxs.append(idx)
		idx += 1

	# Replace label of non-target deposits
	for i in nondeposit_patterns:
		i[-3] = 0

	# Check the number of patterns
	print('Number of deposits', len(deposit_patterns))
	print('Number of non-deposits', len(nondeposit_patterns))

	# Cluster deposits that are close to each others
	# i.e. spatially within a maximum distance cluster_dist
	d_patterns = clusterDep(data=csvData,idxs=deposit_patterns_idxs,dist=cluster_dist)
	nd_patterns = clusterDep(data=csvData,idxs=nondeposit_patterns_idxs,dist=cluster_dist)
	print('Number of deposits (after clustering)',len(d_patterns))
	print('Number of non-deposits (after clustering)',len(nd_patterns))

	return d_patterns, nd_patterns


def add_noise(data,noise):
	'''
	Add extra vectors of data to deposit pattern if dataset too small
	and/or strong class imbalance
	:param n: number of extra (noisy) vectors to add in dataset 
	'''
	# recreated original data with extra n
	mtx = np.zeros(shape=(0,len(data)+noise))

	# same for geology and labels (no random sampling)
	geology = np.ones(shape=(1,len(data)+noise)) * data[0,0]		# permissive geology kept to 1 
	labels = np.ones(shape=(1,len(data)+noise)) * data[0,-3]		# labels kept to 1

	# add extra vectors of data to deposit pattern by random sampling
	for j in range(1,len(data[0,:-3])):
		# normal distribution
		new_data = np.random.normal(np.mean( data[:,j] ),np.std( data[:,j] ),noise)
		column = np.hstack( (data[:,j],new_data) )
		mtx = np.vstack( (mtx, column) )

	mtx = np.vstack( (geology, mtx) )
	mtx = np.vstack( (mtx, labels) ).T

	return mtx

def balance_dataset(dp,ndp,noise=False):
	'''
	Add noise to patterns if class imbalance. Training dataset must have equal d/nd examples.
	'''
	d_diff = len(dp) - len(ndp)
	if d_diff == 0 and noise == True:
		# deactivate noise if data patterns have same size
		noise = False
	elif noise == True:
		noise = np.abs(d_diff)
		if d_diff < 0:
			'''
			Recreate the non-deposit patterns:
			(1) by schuffling the pattern list first
			(2) by sampling a number of nd_patterns = to the number of dp
			this is to avoid having preferential learning by the classifier

			Notes:
				add_noise() rules out the (x,y) coordinates
			'''
			d_patterns_n = add_noise(data=np.asarray(dp),noise=noise)
			nd_patterns_n = np.random.permutation(ndp)
			# [:,:-2] rule out the (x,y) coordinates
			nd_patterns_n = nd_patterns_n[:len(d_patterns_n),:-2]
			#
		else:
			nd_patterns_n = add_noise(data=np.asarray(ndp),noise=noise)
			d_patterns_n = np.random.permutation(dp)
			d_patterns_n = d_patterns_n[:len(nd_patterns_n),:-2]
	else:
		# if no noise
		d_patterns_n = dp[:,:-2]
		nd_patterns_n = ndp[:,:-2]

	# replace labels of non-deposit pattern to 0
	# Note: this is important for transform_labels() in get_inputs() from the model() object
	nd_patterns_n[:,-1] = 0

	return d_patterns_n, nd_patterns_n

def reorganize(ft,lb,test_pct=10,val_pct=10):
	'''
	Dispatch training/testing data
	:param ft:			features
	:param lb:			labels
	:param test_pct:	percent data for testing
	:param val_pct:		percent data for validation
	'''
	# Shuffle
	idx_d = list(range(len(ft[:,0])))					# Data: [_____________]
	np.random.shuffle(idx_d)
	
	# Threshold for test values
	trsh = int((len(ft[:,0]) * test_pct) / 100)	# Portion of data taken: |---/___________|

	# Threshold for validation values
	lgth = int((len(ft[:,0]) * val_pct) / 100)		
	trsh_2 = int( len(ft[:,0]) - lgth )					# |---/________/...|
	
	# Re-organize data in different sets
	new_train_f, new_test_f, new_val_f = [], [], []
	new_train_l, new_test_l, new_val_l = [], [], []
	for i in range(len(ft[0])):
		new_test_f.append( ft[:,i][idx_d][0:trsh] )
		new_train_f.append( ft[:,i][idx_d][trsh:trsh_2] )
		new_val_f.append( ft[:,i][idx_d][trsh_2:-1] )

	new_test_l.append( lb[idx_d][0:trsh] )
	new_train_l.append( lb[idx_d][trsh:trsh_2] )
	new_val_l.append( lb[idx_d][trsh_2:-1] )
	
	new_test_f = np.asarray(new_test_f).T
	new_train_f = np.asarray(new_train_f).T
	new_val_f = np.asarray(new_val_f).T

	new_test_l = np.asarray(new_test_l).T
	new_train_l = np.asarray(new_train_l).T
	new_val_l = np.asarray(new_val_l).T
	
	return new_train_f, new_train_l, new_test_f, new_test_l, new_val_f, new_val_l

def transform_labels(*args):
	'''
	Transform labels into hot-one vectors
	'''
	m = []
	if args is not None:
		for lst in args:
			v = np.array([e/e if e != 0 else 0 for e in lst]).astype(int)
			v = np.eye(2)[v.reshape(-1)].astype(int)
			m.append(v)
	return (i for i in m)

def get_inputs(dp,ndp):
	'''
	Dispatching of training/testing data with homogeneous number of training/testing data
	:param test_pct:	percent data for testing
	:param val_pct:		percent data for validation
	'''
	dp_train_f, dp_train_l, dp_test_f, dp_test_l, dp_val_f, dp_val_l = reorganize(ft=dp[:,:-1],lb=dp[:,-1])
	ndp_train_f, ndp_train_l, ndp_test_f, ndp_test_l, ndp_val_f, ndp_val_l = reorganize(ft=ndp[:,:-1],lb=ndp[:,-1])
	
	# combine the training/testing data
	train_f = np.concatenate((dp_train_f,ndp_train_f),axis=0)
	train_l = np.concatenate((dp_train_l,ndp_train_l),axis=0).flatten()
	test_f = np.concatenate((dp_test_f,ndp_test_f),axis=0)
	test_l = np.concatenate((dp_test_l,ndp_test_l),axis=0).flatten()
	val_f = np.concatenate((dp_val_f,ndp_val_f),axis=0)
	val_l = np.concatenate((dp_val_l,ndp_val_l),axis=0).flatten()

	# get one-hot vectors
	train_l_v, val_l_v, test_l_v = transform_labels(train_l, val_l, test_l)		# One-hot vector transformation
	#
	return train_f, val_f, test_f, train_l_v, val_l_v, test_l_v