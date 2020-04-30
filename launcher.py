import os
os.chdir("C:/Users/cyrilj/Desktop/github-RBFNN")
import utils, model

# Read csv data
path = "./data/"
file = "example.csv"
csvData = utils.read_csv(filename=path+file)

# Convert labels into ternary form (0: no dep.; 1: target dep.; 2: dep.)
# Note: convertion rules out the first column (oid)
csvCopy = utils.convert_labels(csvData=csvData)

# Convert rock units into binary form (0: non-favorable; 1: favorable)
csvProcessed = utils.convert_rock_units(runits=csvCopy)

# Get deposit and non-deposit fingerprints
d_patterns, nd_patterns = utils.get_fingerprints(csvData=csvProcessed,cluster_dist=250)
d_patterns_b, nd_patterns_b = utils.balance_dataset(dp=d_patterns,ndp=nd_patterns,noise=False)

# Get model for radial basis function neural network
rbf_model = model.model()

# Get train/test/validation sets
t_f, val_f, tst_f, t_l, val_l, tst_l = utils.get_inputs(dp=d_patterns_b,ndp=nd_patterns_b)

'''
Model parameters:
	:param n_protos: 		number of prototypes (or hidden neurons) for radial basis function neural network
	:param noise: 			number of extra data added to deposit patterns if dataset too small or class imbalance
	:param test_pct: 		percent of data taken for testing
	:param val_pct: 		percent of data taken for validation
	:param nbatch: 			batch size
	:param outv: 			size of the output layer | should be 2 for "non-deposits" and "deposits" prediction
	:param epochs_nb: 		number of training epochs
	:param lrate: 			learning rate
	:param decay: 			learning decay | lr = lr * (1/(1 + decay * epoch))
	:param reg_factor:		l2 regularization factor on weights
	:param save_path:		path for saving the model parameters and metrics
	:param results_path: 	path for saving predictions
'''

# Train the model
rbf_model.train(train_f=t_f,val_f=val_f,train_l_v=t_l,val_l_v=val_l)

# Test the model
rbf_model.test(test_f=tst_f,test_l_v=tst_l,nproto=50,step=98)

'''
Get predictions:
	:output yp_b: binary outputs "non-deposits" (0), positive for "deposits" (1)
	:output yp_max: maximum probability no matter the output type
	:output yp_c: probability for "non-deposits" (negative) or positive for "deposits" (positive) | to be used for mapping in ArcGIS
'''
yp_b, yp_max, yp_c = rbf_model.predict(features=csvProcessed[:50,:-3],nproto=50,step=98)