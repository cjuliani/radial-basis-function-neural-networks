import tensorflow as tf
import numpy as np
import random, os


class Queue:
	'''
	Queuing system to collect training data randomly and continuously
	'''
	def __init__(self):
		self.items = []

	def isEmpty(self):
		return self.items == []

	def enqueue(self, item):
		self.items.insert(0,item)

	def dequeue(self,n,queue_,features_,labels_):
		try:
			return self.items.pop()
		except:
			generator(ft=features_,lb=labels_,queue=queue_,n=n)
			return self.items.pop()

	def size(self):
		return len(self.items)

def generator(ft,lb,queue,n):
	'''
	Data generator with enqueue system and shuffling
	:param ft: features
	:param lb: labels
	:param queue: queueing system
	:param n: batch size
	'''
	idxs = [i for i in range(len(ft[:,0]))]
	np.random.shuffle(idxs)
	
	ft_shuffled = ft[idxs]
	lt_shuffled = lb[idxs]

	# re-feed shuffled features/labels to the queue
	for i in range(0, len(idxs), n):
		try:
			queue.enqueue([ ft_shuffled[i:i+n], lt_shuffled[i:i+n] ])
		except:
			# if list end reached, gives remaining samples
			queue.enqueue([ ft_shuffled[i:-1], lt_shuffled[i:-1] ])

class model(object):
	'''
	RBFNN implementation
	'''
	def __init__(self):
		# reset default graph
		tf.compat.v1.reset_default_graph()

		# get parameters
		self.n_protos = 50		# number of prototypes / hidden neurons
		self.nbatch = 1			# batch size
		self.outv = 2			# size of the output layer
		self.epochs_nb = 150	# number of training epochs
		self.lrate = 0.5		# learning rate
		self.decay = 0.1		# learning decay | lr = lr * (1/(1 + decay * epoch)); if decay = 0, then no decay is applied
		self.show_n = 5			# display training result after n steps
		self.reg_factor = 0.5	# regularization factor
		self.save_path = './train/'
		self.results_path = './results/'

		with tf.name_scope('Placeholders'):
			self.X = tf.placeholder(shape=[None, None], dtype=tf.float32, name='X_input')			# shape = (None,variables)
			self.Y = tf.placeholder(shape=[None, self.outv], dtype=tf.float32, name='Y_output')		# shape = (? protos, ? outputs)
			self.hOut = tf.placeholder(shape=[None, self.n_protos], dtype=tf.float32, name='hOut')	# shape = (? batches, ? protos)
			self.global_epoch = tf.Variable(0., trainable=False)
		
		with tf.name_scope('Parameters'):
			self.w = tf.Variable(tf.random_normal(shape=[self.n_protos,self.outv]),name='w_hidden_1')		# shape = (? protos, ? outputs)
			self.b = tf.Variable(tf.random_normal(shape=[self.outv]),name='b_hidden_1')						# shape = (? outputs,)

	def progression(self,cnt,data):
		lgth = len(data)/10
		if cnt == 0:
			print("0% done...")
		elif cnt == int( lgth ):
			print("10% done...")
		elif cnt == int( 2 * lgth ) :
			print("20% done...")
		elif cnt == int( 3 * lgth ) :
			print("30% done...")
		elif cnt == int( 4 * lgth ) :
			print("40% done...")
		elif cnt == int( 5 * lgth ) :
			print("50% done...")
		elif cnt == int( 6 * lgth ) :
			print("60% done...")
		elif cnt == int( 7 * lgth ) :
			print("70% done...")
		elif cnt == int( 8 * lgth ) :
			print("80% done...")
		elif cnt == int( 9 * lgth ) :
			print("90% done...")
		elif cnt == int( 10 * lgth ) - 2 :
			print("100% done...")

	def prototyping(self,ft,lb):
		'''
		Select prototypes by including reasonable portion of classes
		~half of prototypes are 1s (deposit patterns)
		:param ft:	features
		:param lb:	labels
		'''
		thresh = int(self.n_protos/2)
		while True:
			idxs_protos = random.sample(range(len(ft)), self.n_protos)
			if np.sum(lb[idxs_protos][:,1]) == thresh:
				break
			else:
				pass
		
		protos = ft[idxs_protos]
		return protos

	def sigma(self,ft,protos):
	    '''
	    Spread parameter
	    :param ft:		features
	    :param protos:	prototypes
	    '''
	    s_protos = []
	    cnt = 0
	    for m in range(0,self.n_protos):
	    	self.progression(cnt=cnt,data=protos)
	    	
	    	dTemp = 0
	    	for i in range(0,ft.shape[0]):
	    		dist = np.square(np.linalg.norm(protos[m] - ft[i]))
	    		dTemp += dist
	    	
	    	# sigma = (max distance between any 2 prototypes)/square_root(total prototypes)
	    	# s = (1/n) * SUM( xi - vi ) for i = 1,2,...,n
	    	s_protos.append(np.sqrt(dTemp)/ft.shape[0])
	    	
	    	cnt = cnt + 1
	    return s_protos

	def network(self):
		'''
		Create the RBF network
		:param X: placeholder for inputs
		:param Y: placeholder for labels
		:param reg_factor: regularizing factor
		'''
		l1_logits = tf.add(tf.matmul(tf.cast(self.hOut,tf.float32),self.w),self.b)

		layer_1 = tf.nn.softmax(l1_logits)
		
		# MSE cost
		cost = tf.losses.mean_squared_error(predictions=layer_1,labels=self.Y)
		#cost = tf.nn.softmax_cross_entropy_with_logits(logits=l1_logits,labels=self.Y, name='cost')

		# Regularized cost
		reg_cost = self.reg_factor * tf.nn.l2_loss(self.w)
		cost = tf.add(tf.reduce_mean( cost ),reg_cost)

		Y_ = tf.argmax(self.Y, axis=1)
		Y_p = tf.argmax(layer_1, axis=1)

		return cost, Y_, Y_p, layer_1

	def metrics(self,y_,y_p):
		'''
		Metrics: recall and specificity
		:param y_: ground truth {0,1}
		:param Y_p: predictions {0,1}
		'''
		TP = tf.compat.v1.count_nonzero( y_p * y_, dtype=tf.float32, axis=0)
		TN = tf.compat.v1.count_nonzero( (y_p - 1) * (y_ - 1), dtype=tf.float32, axis=0)
		FP = tf.compat.v1.count_nonzero( y_p * (y_ - 1), dtype=tf.float32, axis=0)
		FN = tf.compat.v1.count_nonzero( (y_p - 1) * y_, dtype=tf.float32, axis=0)
		# divide_no_NAN in case no TP exist in sample
		rec = tf.math.divide_no_nan( TP, (TP+FN) )
		spec = tf.math.divide_no_nan( TN, (TN+FP) )
		acc = tf.math.divide_no_nan( (TP+TN), (TP+TN+FP+FN) )
		
		# False positive rate
		FPR = tf.math.subtract( tf.constant(1,dtype='float32'), spec )

		return acc, rec, FPR

	def RBF(self,x_,protos,spread):
		'''
		Radial basis function
		:param x_: input x
		:param protos: prototypes
		:param spread: spread parameter
		'''
		hiddenOut = np.zeros(shape=(0,len(protos)))
		for item in x_:
			out=[]
			for proto,sigma in zip(protos,spread):
				distance = np.square(np.linalg.norm(item - proto))
				neuronOut = np.exp(-(distance)/(np.square(sigma)))
				out.append(neuronOut)
			hiddenOut = np.vstack([hiddenOut,np.array(out)])
		return hiddenOut

	def learning_rate(self,epoch):
		'''
		Time-based learning rate schedule
		'''
		lrate = self.lrate * (1/(1 + self.decay * epoch))
		return lrate


	def test(self,test_f,test_l_v,nproto,step):
		'''
		Test the model with trained parameters
		'''
		# Import trained parameters
		w = np.load(self.save_path+'weights_nproto-'+str(nproto)+'step-'+str(step)+'.npy')
		b = np.load(self.save_path+'bias_nproto-'+str(nproto)+'step-'+str(step)+'.npy')
		protos_f = np.load(self.save_path+'prototypes_nproto-'+str(nproto)+'step-'+str(step)+'.npy')
		spread = np.load(self.save_path+'spread_nproto-'+str(nproto)+'step-'+str(step)+'.npy')

		# Calculate logits
		hOut = self.RBF(x_=test_f,protos=protos_f,spread=spread)
		logits = tf.add(tf.nn.softmax(tf.matmul(tf.cast(hOut,tf.float32), w)), b)
		logits = tf.nn.softmax(logits)
		y_  = tf.argmax(test_l_v, axis=1)
		yp_ = tf.argmax(logits, axis=1)

		# Get metrics
		cost = tf.losses.mean_squared_error(predictions=logits,labels=test_l_v)
		auc = tf.metrics.auc(labels=y_,predictions=tf.math.reduce_max(input_tensor=logits,axis=1))
		acc, TPR, FPR = self.metrics(y_=y_,y_p=yp_)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())

			cost_ = sess.run(cost)
			auc_ = sess.run(auc)
			TPR_ = sess.run(TPR)
			FPR_ = sess.run(FPR)
			acc_ = sess.run(acc)

		print('Cost: ', "%.2f" % cost_, 'Accuracy: ', "%.2f" % acc_, 'TPR: ', "%.2f" % TPR_, 'FPR: ', "%.3f" % FPR_, 'AUC: ', "%.2f" % auc_[0])

	def train(self,train_f, val_f, train_l_v, val_l_v):
		'''
		Get training/testing/validation sets
		:param f: data vector
		:param l_v: one-hot vector
		'''
		# get training/validation/test sets
		data_f = np.concatenate((train_f,val_f),axis=0)
		data_l = np.concatenate((train_l_v,val_l_v),axis=0)
		
		# get prototypes and spread parameter
		print('Prototyping...')
		protos_f = self.prototyping(ft=data_f,lb=data_l)
		spread = self.sigma(ft=data_f,protos=protos_f)
		print('Done.')

		# Get number oft training steps given batch size
		steps = int(len(train_f) / self.nbatch)
		steps_t = int(len(val_f))

		print('\nEpochs: ', self.epochs_nb)
		print('Prototypes: ', self.n_protos)
		print('Learning rate: ', self.lrate)
		print('Reg. factor: ', self.reg_factor)
		print('\n')
		print("Training set: ", len(train_f))
		print("Validation set: ", len(val_f))
		print("Testing set: ", len(test_f))
		print('\n')
		print("Batches: ", self.nbatch)
		print("Steps (training): ", steps)
		print("Steps (validation): ", steps_t)
		print('\n')

		# Open file to write results
		# Note: this is to overwrite old files
		__ = open(self.save_path+'training.txt', 'w')
		__ = open(self.save_path+'validation.txt', 'w')

		with tf.Session() as sess:

			# get metrics and predictions
			cost, y_, yp_, prob_ = self.network()
			auc = tf.metrics.auc(labels=y_,predictions=tf.math.reduce_max(input_tensor=prob_,axis=1))
			a, TPR, FPR = self.metrics(y_=y_,y_p=yp_)

			# get learning rate and optimizer
			lr_ = self.learning_rate(epoch=self.global_epoch)
			optimizer = tf.train.GradientDescentOptimizer(lr_).minimize(cost)
			#optimizer = tf.train.AdamOptimizer(lr_).minimize(cost)

			# initialize network variables
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())

			cnt = 0		# counter for displaying training progressions
			for epoch in range(self.epochs_nb):
				# re-initialize metrics at every epoch
				avg_cost, avg_acc, avg_auc, avg_TPR, avg_FPR = 0.,0.,0.,0.,0.
				avg_c_valid,avg_a_valid,avg_auc_valid,avg_TPR_v,avg_FPR_v = 0.,0.,0.,0.,0.

				#probs_range = []

				# generate training data
				q = Queue()
				generator(ft=train_f,lb=train_l_v,queue=q,n=self.nbatch)
				
				# generate validation data
				z = Queue()
				generator(ft=val_f,lb=val_l_v,queue=z,n=self.nbatch)
				
				lrate = sess.run(lr_, feed_dict={self.global_epoch:epoch})
				
				# Loop over all batches of size n
				for step in range(steps):
					
					# get training data
					ft_train = q.dequeue(n=self.nbatch,queue_=q,features_=train_f,labels_=train_l_v)
					aX = ft_train[0]	# features
					aY = ft_train[1]	# labels
					
					# get hidden neurons outputs
					hOut_t = self.RBF(x_=aX,protos=protos_f,spread=spread)

					input_dict = {self.hOut:hOut_t,self.Y:aY}
					sess.run(optimizer, feed_dict=input_dict)
					
					# get predictions and metrics
					c = sess.run(cost, feed_dict=input_dict)
					auC = sess.run(auc, feed_dict=input_dict)
					#p_ = sess.run(prob_, feed_dict=input_dict)
					#maxp_ = list( np.max(p_, axis=1) )

					TPR_ = sess.run(TPR, feed_dict=input_dict)
					FPR_ = sess.run(FPR, feed_dict=input_dict)
					acc_ = sess.run(a, feed_dict=input_dict)
					
					# Compute average cost & accuracy
					avg_cost += (c / steps)
					avg_acc += (acc_ / steps)
					avg_auc +=(auC[0] / steps)
					avg_TPR += (TPR_ / steps)
					avg_FPR += (FPR_ / steps)
					# probs_range += probs_range + maxp_

					#if (step % self.show_n == 0) or (step == 0):
					#	msg = "Epoch {0} / {1} --- Cost: {2:.3f} --- Accuracy: {3:.3f} --- TPR: {4:.3f} --- FPR: {5:.3f} --- AUC: {6:.3f}"
					#	print(msg.format(epoch, self.epochs_nb, c, a, TPR, FPR, auC[0]))

					# Save model's parameters when an epoch starts or ends
					if (step == 0) or (step == (steps - 1)):
						np.save(self.save_path+'weights_nproto-'+str(self.n_protos)+'step-'+str(step), self.w.eval())
						np.save(self.save_path+'bias_nproto-'+str(self.n_protos)+'step-'+str(step), self.b.eval())
						np.save(self.save_path+'prototypes_nproto-'+str(self.n_protos)+'step-'+str(step), protos_f)
						np.save(self.save_path+'spread_nproto-'+str(self.n_protos)+'step-'+str(step), spread)
				
				'''
				Validation
				'''
				for i in range(steps_t):
					# get data
					ft_val = z.dequeue(n=self.nbatch,queue_=z,features_=val_f,labels_=val_l_v)
					bX = ft_val[0]
					bY = ft_val[1]
					
					hOut_v = self.RBF(x_=bX,protos=protos_f,spread=spread)
					input_dict_v = {self.hOut:hOut_v,self.Y:bY}
					
					# get metrics
					cost_v = sess.run(cost, feed_dict=input_dict_v)
					auc_valid = sess.run(auc, feed_dict=input_dict_v)
					TPR_v = sess.run(TPR, feed_dict=input_dict_v)
					FPR_v = sess.run(FPR, feed_dict=input_dict_v)
					acc_v = sess.run(a, feed_dict=input_dict_v)
					
					avg_c_valid += (cost_v / steps_t)
					avg_a_valid += (acc_v / steps_t)
					avg_auc_valid += (auc_valid[0] / steps_t)
					avg_TPR_v += (TPR_v / steps_t)
					avg_FPR_v += (FPR_v / steps_t)

				if (epoch % self.show_n == 0) or (step == 0):
					# display training results
					msg = "Epoch {0} / {1} --- Cost: {2:.3f} --- Accuracy: {3:.2f} --- TPR: {4:.3f} --- FPR: {5:.3f} --- AUC: {6:.3f}"
					print(msg.format(epoch, self.epochs_nb, avg_cost, avg_acc, avg_TPR, avg_FPR, avg_auc))

					# display validation results
					msg = "Validation --- Cost: {0:.3f} --- Accuracy: {1:.2f} --- TPR: {2:.3f} --- FPR: {3:.3f} --- AUC: {4:.3f}"
					print(msg.format(avg_c_valid, avg_a_valid, avg_TPR_v, avg_FPR_v, avg_auc_valid))
					print('\n')

				# Write metrics
				cont = 'Epoch ' + str(epoch) + "; cost: " + str(avg_cost) + "; accuracy: " + str(avg_acc) + "; TPR: " + str(avg_TPR) + "; FPR: " + str(avg_FPR) + "; AUC: " + str(avg_auc) + "\n"
				# Append new lines to opened text file
				# Note: append 'a' is more stable than write 'w' during the training phase
				with open(self.save_path+'training.txt', 'a') as file_a:
					file_a.write(cont)

				cont_val = 'Epoch ' + str(epoch) + "; cost: " + str(avg_c_valid) + "; accuracy: " + str(avg_a_valid) + "; TPR: " + str(avg_TPR_v) + "; FPR: " + str(avg_FPR_v) + "; AUC: " + str(avg_auc_valid) + "\n"
				with open(self.save_path+'validation.txt', 'a') as file_b:
					file_b.write(cont_val)

	def predict(self,features,nproto=50,step=98):
		'''
		Predict probability of getting either {non-deposit,deposit} at a location
		'''

		# Import train parameters
		w = np.load(self.save_path+'weights_nproto-'+str(nproto)+'step-'+str(step)+'.npy')
		b = np.load(self.save_path+'bias_nproto-'+str(nproto)+'step-'+str(step)+'.npy')
		protos_f = np.load(self.save_path+'prototypes_nproto-'+str(nproto)+'step-'+str(step)+'.npy')
		spread = np.load(self.save_path+'spread_nproto-'+str(nproto)+'step-'+str(step)+'.npy')

		# Calculate logits
		hOut = self.RBF(x_=features,protos=protos_f,spread=spread)
		logits = tf.add(tf.nn.softmax(tf.matmul(tf.cast(hOut,tf.float32), w)), b)
		logits = tf.nn.softmax(logits)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())

			yp_ = list( sess.run( tf.argmax(logits, axis=1) ) )								# {0,1} or {non-deposit,deposit}
			yp_max = list( sess.run( tf.math.reduce_max(input_tensor=logits,axis=1) ) )		# max probability values, no matter the output type

			# if non-deposit (0), assign negative probability value, else assign positive prob. if deposit (1)
			yp_c = np.asarray([ yp_max[i] * -1 if yp_[i] == 0 else yp_max[i] for i in range(len(yp_)) ])

		return yp_, yp_max, yp_c