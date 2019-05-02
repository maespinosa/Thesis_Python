import numpy as np


import sys
sys.path.append('C:/Users/Marks-M3800/Documents/A2/assignment2')

#from cs231n.layers import *
#from cs231n.fast_layers import *
#from cs231n.layer_utils import *
from tf_layers import *
sys.path.append('C:/Anaconda3/envs')
import tensorflow as tf
import os
import struct


class AlexNet_TF(object):
	def __init__(self): 
		pass

	def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
	    '''From https://github.com/ethereon/caffe-tensorflow
	    '''
	    c_i = input.get_shape()[-1]
	    assert c_i%group==0
	    assert c_o%group==0
	    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
	    
	    
	    if group==1:
	        conv = convolve(input, kernel)
	    else:
	        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
	        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
	        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
	        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
	    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



	def affine_forward_tf(X,W,b): 

		print('X shape = ', X.shape)
		print('W shape = ', W.shape)
		print('b shape = ', b.shape)
		x_reshape = tf.reshape(X,(X.shape[0],tf.reduce_prod(X.shape[1:])))
		print('x_reshape shape = ', x_reshape.shape)


		out = tf.tensordot(tf.transpose(W), tf.transpose(x_reshape), axes=((1),(0))) 

		print('out shape = ', out.shape)

		if(out.shape[1] != 1): 

			b_tiled = tf.tile(b,[int(out.shape[1])])
			b = tf.reshape(b_tiled, (b.shape[0], out.shape[1]))
			print('new b shape = ', b.shape)

		out = tf.add(out,b)
		return out

	def get_saved_parameters(self, hidden_dim = 4096, num_classes = 10): 

		self.params = {}
		## W1 Init ---------------------------------------------------
		
		file_exists = os.path.isfile('./params_temp/W1.bin')
		file_size = os.path.getsize('./params_temp/W1.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('W1.bin FOUND!!')
		    file = open('./params_temp/W1.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    data = np.reshape(data, [96,3,11,11])
		    self.params['W1'] = np.transpose(data, [2,3,1,0])

		
		## b1 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/b1.bin')
		file_size = os.path.getsize('./params_temp/b1.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('b1.bin FOUND!!')
		    file = open('./params_temp/b1.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['b1'] = np.reshape(data, [96])


		## W2 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/W2.bin')
		file_size = os.path.getsize('./params_temp/W2.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('W2.bin FOUND!!')
		    file = open('./params_temp/W2.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['W2'] = np.reshape(data, [256,96,5,5])


		## b2 Init ---------------------------------------------------			
		file_exists = os.path.isfile('./params_temp/b2.bin')
		file_size = os.path.getsize('./params_temp/b2.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('b2.bin FOUND!!')
		    file = open('./params_temp/b2.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['b2'] = np.reshape(data, [256])
			
		## W3 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/W3.bin')
		file_size = os.path.getsize('./params_temp/W3.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('W3.bin FOUND!!')
		    file = open('./params_temp/W3.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['W3'] = np.reshape(data, [384,256,3,3])
		
		
		## b3 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/b3.bin')
		file_size = os.path.getsize('./params_temp/b3.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('b3.bin FOUND!!')
		    file = open('./params_temp/b3.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['b3'] = np.reshape(data, [384])
			
			
		## W4 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/W4.bin')
		file_size = os.path.getsize('./params_temp/W4.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('W4.bin FOUND!!')
		    file = open('./params_temp/W4.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['W4'] = np.reshape(data, [384,384,3,3])
		
		
		
		## b4 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/b4.bin')
		file_size = os.path.getsize('./params_temp/b4.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('b4.bin FOUND!!')
		    file = open('./params_temp/b4.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['b4'] = np.reshape(data, [384])
			
			
			
		## W5 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/W5.bin')
		file_size = os.path.getsize('./params_temp/W5.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('W5.bin FOUND!!')
		    file = open('./params_temp/W5.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['W5'] = np.reshape(data, [256,384,3,3])
		

		
		## b5 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/b5.bin')
		file_size = os.path.getsize('./params_temp/b5.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('b5.bin FOUND!!')
		    file = open('./params_temp/b5.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['b5'] = np.reshape(data, [256])
			
			
		## W6 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/W6.bin')
		file_size = os.path.getsize('./params_temp/W6.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('W6.bin FOUND!!')
		    file = open('./params_temp/W6.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['W6'] = np.reshape(data, [256*6*6,hidden_dim])
		
		
		## b6 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/b6.bin')
		file_size = os.path.getsize('./params_temp/b6.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('b6.bin FOUND!!')
		    file = open('./params_temp/b6.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['b6'] = np.reshape(data, [hidden_dim])
			
			
		## W7 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/W7.bin')
		file_size = os.path.getsize('./params_temp/W7.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('W7.bin FOUND!!')
		    file = open('./params_temp/W7.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['W7'] = np.reshape(data, [hidden_dim,hidden_dim])
		
		## b7 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/b7.bin')
		file_size = os.path.getsize('./params_temp/b7.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('b7.bin FOUND!!')
		    file = open('./params_temp/b7.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['b7'] = np.reshape(data, [hidden_dim])
			
			
		## W8 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/W8.bin')
		file_size = os.path.getsize('./params_temp/W8.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('W8.bin FOUND!!')
		    file = open('./params_temp/W8.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['W8'] = np.reshape(data, [hidden_dim,num_classes])
		
		
		## b8 Init ---------------------------------------------------
		file_exists = os.path.isfile('./params_temp/b8.bin')
		file_size = os.path.getsize('./params_temp/b8.bin')

		data = np.zeros([int(file_size/8)])
		if file_exists:
		    print('b8.bin FOUND!!')
		    file = open('./params_temp/b8.bin','rb')
		    for i in range(int(file_size/8)):
		        data[i] = struct.unpack('d', file.read(8))[0]
		    file.close()

		    self.params['b8'] = np.reshape(data, [num_classes])
			



	def loss(self,X, y=None, input_dim = (1, 3, 227, 227), hidden_dim = 4096, num_classes = 1000, reg=0.0, weight_scale = 1e-3, get_saved_params = 0):

		if(get_saved_params == 1): 
			self.get_saved_parameters(hidden_dim = hidden_dim, num_classes = num_classes)


		conv1_param = {'stride': 4, 'pad': 0}
		conv2_param = {'stride': 1, 'pad': 2}
		conv3_param = {'stride': 1, 'pad': 1}
		conv4_param = {'stride': 1, 'pad': 1}
		conv5_param = {'stride': 1, 'pad': 1}				

		# pass pool_param to the forward pass for the max-pooling layer
		pool_param = {'pool_height': 3, 'pool_width': 3, 'stride': 2}

		#X = tf.placeholder(tf.float32, shape = (input_dim))
		#y = tf.placeholder(tf.int32, shape = (input_dim[0]))
		#loss = tf.placeholder(tf.float32)

		# print('X shape = ', X.shape)
		# print('W1 shape = ', W1.shape)

		X1 = tf.subtract(X,tf.reduce_mean(X,axis=0))
		X2 = tf.math.reduce_std(X1,axis=0)
		X3 = tf.divide(X1,X2)


		with tf.variable_scope("CONV1"):
			#print('>>> CONV1')
			file_exists = os.path.isfile('./params/W1.ckpt.data-00000-of-00001')
			file_size = os.path.getsize('./params/W1.ckpt.data-00000-of-00001')

			if(get_saved_params == 1 & file_exists == 1): 
				#W1 = tf.get_variable("W1", [11,11,3,96], initializer = tf.zeros_initializer)
				#saver = tf.train.Saver({"W1": W1})
				#W1.initializer.run()
				#saver.restore(sess, './params/W1.ckpt.data-00000-of-00001')
				W1 = tf.convert_to_tensor(self.params['W1'], dtype=tf.float32)
				pass
			else: 
				W1 = tf.Variable(tf.random_normal((11,11,3,96), mean=0, stddev=weight_scale))


			b1 = tf.Variable(tf.zeros(96))
			conv1 = tf.Variable(tf.random_normal((input_dim[0],96,55,55), mean=0, stddev=weight_scale))

			conv1_out = tf.assign(conv1,tf.nn.conv2d(input=X3,filter=W1,strides=[1,1, conv1_param['stride'], conv1_param['stride']],padding='VALID',data_format="NCHW"))
			#conv1_out = tf.assign(conv1,conv_forward_naive(x=X,w=W1,b=b1,conv_param=conv1_param))
			#conv1_out = tf.assign(conv1,AlexNet_TF.conv(input=X, kernel=W1, biases=b1, k_h=11, k_w=11, c_o=96, s_h=4, s_w=4,  padding="VALID", group=1))

			print('CONV1 OUT shape = ', conv1_out.shape)


		with tf.variable_scope("RELU1"):
			#print('>>> RELU1')
			relu1 = tf.Variable(tf.random_normal((input_dim[0],96,55,55), mean=0, stddev=weight_scale))
			relu1_out = tf.assign(relu1, tf.nn.relu(features=conv1))
			print('RELU1 OUT shape = ', relu1_out.shape)

		with tf.variable_scope("MAXPOOL1"):
			#print('>>> MAXPOOL1')
			maxpool1 = tf.Variable(tf.random_normal((input_dim[0],96,27,27), mean=0, stddev=weight_scale))
			max_pool1_out = tf.assign(maxpool1, tf.nn.max_pool(value=relu1, ksize=[1,1,pool_param['pool_height'], pool_param['pool_width']],strides=[1,1,pool_param['stride'], pool_param['stride']], padding= 'VALID',data_format='NCHW'))
			print('MAXPOOL1 OUT shape = ', max_pool1_out.shape)


		with tf.variable_scope("CONV2"):
			#print('>>> CONV2')
			W2 = tf.Variable(tf.random_normal((5,5,96,256), mean=0, stddev=weight_scale))
			b2 = tf.Variable(tf.zeros(256))
			conv2 = tf.Variable(tf.random_normal((input_dim[0],256,27,27), mean=0, stddev=weight_scale))
			
			conv2_out = tf.assign(conv2,tf.nn.conv2d(input=maxpool1,filter=W2,strides=[1,1, conv2_param['stride'], conv2_param['stride']],padding='SAME', data_format="NCHW"))
			print('CONV2 OUT shape = ', conv2_out.shape)
		
		# #print('>>> RELU2')
		# relu2_out = tf.nn.relu(features=conv2_out)
		# print('RELU2 OUT shape = ', relu2_out.shape)
		# #print('>>> MAXPOOL2')
		# max_pool2_out = tf.nn.max_pool(value=relu2_out, ksize=[1,1,pool_param['pool_height'], pool_param['pool_width']],strides=[1,1,pool_param['stride'], pool_param['stride']], padding= 'VALID',data_format='NCHW')
		# print('MAXPOOL2 OUT shape = ', max_pool2_out.shape)
		
		with tf.variable_scope("RELU2"):
			#print('>>> RELU2')
			relu2 = tf.Variable(tf.random_normal((input_dim[0],256,27,27), mean=0, stddev=weight_scale))
			relu2_out = tf.assign(relu2, tf.nn.relu(features=conv2))
			print('RELU2 OUT shape = ', relu2_out.shape)

		with tf.variable_scope("MAXPOOL2"):
			#print('>>> MAXPOOL2')
			maxpool2 = tf.Variable(tf.random_normal((input_dim[0],256,13,13), mean=0, stddev=weight_scale))
			max_pool2_out = tf.assign(maxpool2, tf.nn.max_pool(value=relu2, ksize=[1,1,pool_param['pool_height'], pool_param['pool_width']],strides=[1,1,pool_param['stride'], pool_param['stride']], padding= 'VALID',data_format='NCHW'))
			print('MAXPOOL2 OUT shape = ', max_pool2_out.shape)


		with tf.variable_scope("CONV3"):
			#print('>>> CONV3')
			W3 = tf.Variable(tf.random_normal((3,3,256,384), mean=0, stddev=weight_scale))
			b3 = tf.Variable(tf.zeros(384))
			conv3 = tf.Variable(tf.random_normal((input_dim[0],384,13,13), mean=0, stddev=weight_scale))
			
			conv3_out = tf.assign(conv3,tf.nn.conv2d(input=maxpool2,filter=W3,strides=[1,1, conv3_param['stride'], conv3_param['stride']],padding='SAME',data_format="NCHW"))
			print('CONV3 OUT shape = ', conv3_out.shape)
		
		# #print('>>> RELU3')
		# relu3_out = tf.nn.relu(features=conv3_out)
		# print('RELU3 OUT shape = ', relu3_out.shape)


		with tf.variable_scope("RELU3"):
			#print('>>> RELU3')
			relu3 = tf.Variable(tf.random_normal((input_dim[0],384,13,13), mean=0, stddev=weight_scale))
			relu3_out = tf.assign(relu3, tf.nn.relu(features=conv3))
			print('RELU3 OUT shape = ', relu3_out.shape)
		
		with tf.variable_scope("CONV4"):
			#print('>>> CONV4')
			W4 = tf.Variable(tf.random_normal((3,3,384,384), mean=0, stddev=weight_scale))
			b4 = tf.Variable(tf.zeros(384))
			conv4 = tf.Variable(tf.random_normal((input_dim[0],384,13,13), mean=0, stddev=weight_scale))

			conv4_out = tf.assign(conv4,tf.nn.conv2d(input=relu3,filter=W4,strides=[1,1, conv4_param['stride'], conv4_param['stride']],padding='SAME',data_format="NCHW"))
			print('CONV4 OUT shape = ', conv4_out.shape)
		
		# relu4_out = tf.nn.relu(features=conv4_out)
		# print('RELU4 OUT shape = ', relu4_out.shape)


		with tf.variable_scope("RELU4"):
			#print('>>> RELU4')
			relu4 = tf.Variable(tf.random_normal((input_dim[0],384,13,13), mean=0, stddev=weight_scale))
			relu4_out = tf.assign(relu4, tf.nn.relu(features=conv4))
			print('RELU4 OUT shape = ', relu4_out.shape)


		with tf.variable_scope("CONV5"):
			#print('>>> CONV5')
			W5 = tf.Variable(tf.random_normal((3,3,384,256), mean=0, stddev=weight_scale))
			b5 = tf.Variable(tf.zeros(256))
			conv5 = tf.Variable(tf.random_normal((input_dim[0],256,13,13), mean=0, stddev=weight_scale))
			
			conv5_out = tf.assign(conv5,tf.nn.conv2d(input=relu4,filter=W5,strides=[1,1, conv5_param['stride'], conv5_param['stride']],padding='SAME',data_format="NCHW"))
			print('CONV5 OUT shape = ', conv5_out.shape)

		# relu5_out = tf.nn.relu(features=conv5_out)
		# print('RELU5 OUT shape = ', relu5_out.shape)
		# #print('>>> MAXPOOL3')
		# max_pool3_out = tf.nn.max_pool(value=relu5_out, ksize=[1,1,pool_param['pool_height'], pool_param['pool_width']],strides=[1,1,pool_param['stride'], pool_param['stride']], padding= 'VALID',data_format='NCHW')
		# print('MAXPOOL3 OUT shape = ', max_pool3_out.shape)

		with tf.variable_scope("RELU5"):
			#print('>>> RELU5')
			relu5 = tf.Variable(tf.random_normal((input_dim[0],256,13,13), mean=0, stddev=weight_scale))
			relu5_out = tf.assign(relu5, tf.nn.relu(features=conv5))
			print('RELU5 OUT shape = ', relu5_out.shape)

		with tf.variable_scope("MAXPOOL3"):
			#print('>>> MAXPOOL3')
			maxpool3 = tf.Variable(tf.random_normal((input_dim[0],256,6,6), mean=0, stddev=weight_scale))
			max_pool3_out = tf.assign(maxpool3, tf.nn.max_pool(value=relu5, ksize=[1,1,pool_param['pool_height'], pool_param['pool_width']],strides=[1,1,pool_param['stride'], pool_param['stride']], padding= 'VALID',data_format='NCHW'))
			print('MAXPOOL3 OUT shape = ', max_pool3_out.shape)


		with tf.variable_scope("FC6"):
			#print('>>> Affine1')
			W6 = tf.Variable(tf.random_normal((256*6*6,hidden_dim), mean=0, stddev=weight_scale))
			b6 = tf.Variable(tf.zeros(hidden_dim,1))
			affine1 = tf.Variable(tf.zeros((hidden_dim,input_dim[0]))) 
			#affine1_out = tf.assign(affine1,AlexNet_TF.affine_forward_tf(maxpool3, W6, b6))
			#print('AFFINE1 OUT shape =', affine1_out.shape)

		with tf.variable_scope("FC7"):
			#print('>>> Affine2')
			W7 = tf.Variable(tf.random_normal((hidden_dim,hidden_dim), mean=0, stddev=weight_scale))
			b7 = tf.Variable(tf.zeros(hidden_dim,1))
			affine2 = tf.Variable(tf.zeros((hidden_dim,input_dim[0]))) 
			#affine2_out = tf.assign(affine2,AlexNet_TF.affine_forward_tf(tf.transpose(affine1), W7, b7))
			#print('AFFINE2 OUT shape =', affine2_out.shape)

		with tf.variable_scope("FC8"):
			#print('>>> Affine3')
			W8 = tf.Variable(tf.random_normal((hidden_dim,num_classes), mean=0, stddev=weight_scale))
			b8 = tf.Variable(tf.zeros(num_classes,1))
			affine3 = tf.Variable(tf.zeros((num_classes,input_dim[0]))) 
			scores = tf.Variable(tf.zeros((num_classes,hidden_dim)))
			#affine3_out = tf.assign(affine3,AlexNet_TF.affine_forward_tf(tf.transpose(affine2), W8, b8))
			#print('AFFINE3 OUT shape =', affine3_out.shape)
			#print('>>> Scores')
			scores = affine3



		print('scores shape = ', scores.shape)

		print('AlexNet Forward Pass Done')


		# if y is None:
		#   return scores

		# softmax_input_trans = tf.Variable(tf.zeros(tf.transpose(scores).shape))
		# exp_scores = tf.Variable(tf.zeros(softmax_input_trans.shape))
		# sum_exp_scores = tf.Variable(tf.zeros(exp_scores.shape))
		# P = tf.Variable(tf.zeros(softmax_input_trans.shape[0]))
		# #Py = tf.Variable(tf.zeros(y.shape))
		# Py = tf.Variable(tf.zeros(y.shape[0]))
		# print('Py shape = ', Py.shape)
		# Py_val =[]


		#print("SOFTMAX ==================")
		#print('Y = ', y)
		#print('Y shape = ', y.shape)
		softmax_input_trans = tf.transpose(scores)
		print('softmax input trans shape = ', softmax_input_trans.shape)
		exp_scores = tf.exp(softmax_input_trans)
		print('exp of scores shape = ', exp_scores.shape)
		sum_exp_scores = tf.reduce_sum(exp_scores, axis = 0)
		print('sum of exp scores shape = ', sum_exp_scores.shape)

		P = tf.divide(exp_scores,sum_exp_scores)
	
		# # for i in range(y.shape[0]): 
		# #     #P[:,i] = tf.divide(exp_scores[:,i],sum_exp_scores[i])
		# #     P = tf.stack([P[:,i-1],tf.divide(exp_scores[:,i],sum_exp_scores[i])])

		# #print("Probabilities = ", P)
		# print('Probabilities shape = ', P.shape)

		# if y.shape[0] == 1: 
		# 	print('y = ', y)
		# 	y_val = y[0,0]
		# 	Py = P[y_val]
		# else: 
		# 	for i in range(y.shape[0]): 
		# 		#print(y[i])
		# 		#print(P[y[i],i].shape)
		# 		y_val = y[i,0]
		# 		print('y_val shape = ', y_val.shape)
		# 		P_val = P[y_val,i]
		# 		print('P_val shape =', P_val.shape)
		# 		print('P_val = ', P_val)

		# 		Py_val.append(P_val)


		# 		#Py[i] = P_val
		# 		pass
		# 	Py = tf.stack(Py_val)




		# #print('Py = ', Py)
		# print('Py shape = ', Py.shape)	

		# data_loss = -1*tf.log(Py)
		# print('data_loss shape =', data_loss.shape)

		# #print('data loss = ', data_loss)
		# #print('reg = ', self.reg)
		# print(softmax_input_trans.shape[1])

		# avg_data_loss =  tf.divide(1.0,tf.cast(softmax_input_trans.shape[1],tf.float32)) * tf.reduce_sum(data_loss)
		# # print('average data loss = ', avg_data_loss)
		# #print('average data loss shape = ', avg_data_loss.shape)

		# reg_loss = 0.5*reg*(tf.reduce_sum(W1**2) + tf.reduce_sum(W2**2)  + tf.reduce_sum(W3**2) + tf.reduce_sum(W4**2) + tf.reduce_sum(W5**2) + tf.reduce_sum(W6**2) + tf.reduce_sum(W7**2) + tf.reduce_sum(W8**2))
		# #print('regularization loss = ', reg_loss)
		# #print('regularization loss = ', reg_loss.shape)

		# loss = avg_data_loss + reg_loss

		# print('loss = ', loss)

		if(y != None):

			print('y shape = ', y.shape)
			print('y = ', y)
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= y[:,0], logits=tf.transpose(affine3))

			print("Loss ", loss.shape)
			#outputs = [conv1_out,conv2_out,conv3_out,conv4_out,conv5_out]

			dscores = tf.add(P, -1.0)

			dscores = tf.divide(dscores,tf.cast(input_dim[0], tf.float32))
			print('dscores')
		else:
			loss = 0


		# #with tf.variable_scope("BACK_FC8"):
		# 	#dW8 = tf.Variable(tf.random_normal((hidden_dim,num_classes), mean=0, stddev=weight_scale))
		# 	#db8 = tf.Variable(tf.zeros(num_classes,1))
		# dX8 = tf.gradients(dscores, affine2_out)
		# dW8 = tf.gradients(dscores, W8)
		# db8 = tf.gradients(dscores, b8)



		# 	#print('AFFINE3 BACK shape =', dX8)
		# 	#print('FC8 BACK')

		# #with tf.variable_scope("BACK_FC7"):
		# 	#dW7 = tf.Variable(tf.random_normal((hidden_dim,hidden_dim), mean=0, stddev=weight_scale))
		# 	#db7 = tf.Variable(tf.zeros(hidden_dim,1))
		# dX7 = tf.gradients(dX8, affine1_out)
		# dW7 = tf.gradients(dX8, W7)
		# db7 = tf.gradients(dX8, b7)
		# 	#print('AFFINE2 BACK shape =', dX7)
		# 	#print('FC7 BACK')

		# #with tf.variable_scope("BACK_FC6"):
		# 	#dW6 = tf.Variable(tf.random_normal((256*6*6,hidden_dim), mean=0, stddev=weight_scale))
		# 	#db6 = tf.Variable(tf.zeros(hidden_dim,1))
		# dX6 = tf.gradients(dX7, max_pool3_out)
		# dW6 = tf.gradients(dX7, W6)
		# db6 = tf.gradients(dX7, b6)
		# 	#print('AFFINE1 BACK shape =', dX6)
		# 	#print('FC6 BACK')

		# #with tf.variable_scope("BACK_MAXPOOL3"):
		# dMP3 = tf.gradients(dX6, relu5_out)
		# 	#print('MAXPOOL3 BACK shape =', dMP3)
		# 	#print('MAXPOOL3 BACK')

		# #with tf.variable_scope("BACK_RELU5"):
		# dR5 = tf.gradients(dMP3, conv5_out)
		# 	#print('RELU5 BACK shape =', dR5)
		# 	#print('RELU5 BACK')

		# #with tf.variable_scope("BACK_CONV5"):
		# 	#print('>>> CONV5')
		# 	#dW5 = tf.Variable(tf.random_normal((3,3,384,256), mean=0, stddev=weight_scale))
		# 	#db5 = tf.Variable(tf.zeros(256))
		# dX5 = tf.gradients(dR5, relu4_out)
		# dW5 = tf.gradients(dR5, W5)
		# db5 = tf.gradients(dR5, b5)
		# 	#print('CONV5 BACK shape = ', dX5)
		# 	#print('CONV5 BACK')


		# #with tf.variable_scope("BACK_RELU4"):
		# dR4 = tf.gradients(dX5, conv4_out)
		# 	#print('RELU4 BACK shape =', dR4)
		# 	#print('RELU4 BACK')

		# #with tf.variable_scope("BACK_CONV4"):
		# 	#dW4 = tf.Variable(tf.random_normal((3,3,384,384), mean=0, stddev=weight_scale))
		# 	#db4 = tf.Variable(tf.zeros(384))
		# dX4 = tf.gradients(dR4, relu3_out)
		# dW4 = tf.gradients(dR4, W4)
		# db4 = tf.gradients(dR4, b4)
		# 	#print('CONV4 BACK shape = ', dX4)
		# 	#print('CONV4 BACK')



		# #with tf.variable_scope("BACK_RELU3"):
		# dR3 = tf.gradients(dX4, conv3_out)
		# 	#print('RELU3 BACK shape =', dR3)

		# #with tf.variable_scope("BACK_CONV3"):
		# 	#dW3 = tf.Variable(tf.random_normal((3,3,256,384), mean=0, stddev=weight_scale))
		# 	#db3 = tf.Variable(tf.zeros(384))
		# dX3 = tf.gradients(dR3, max_pool2_out)
		# dW3 = tf.gradients(dR3, W3)
		# db3 = tf.gradients(dR3, b3)
		# 	#print('CONV3 BACK shape = ', dX3)

		# #with tf.variable_scope("BACK_MAXPOOL2"):
		# dMP2 = tf.gradients(dX3, relu2_out)
		# 	#print('MAXPOOL2 BACK shape =', dMP2)

		# #with tf.variable_scope("BACK_RELU2"):
		# dR2 = tf.gradients(dMP2, conv2_out)
		# 	#print('RELU2 BACK shape =', dR2)

		# #with tf.variable_scope("BACK_CONV2"):
		# 	#print('>>> CONV5')
		# 	#dW2 = tf.Variable(tf.random_normal((5,5,96,256), mean=0, stddev=weight_scale))
		# 	#db2 = tf.Variable(tf.zeros(256))
		# dX2 = tf.gradients(dR2, relu1_out)
		# dW2 = tf.gradients(dR2, W2)
		# db2 = tf.gradients(dR2, b2)
		# 	#print('CONV2 BACK shape = ', dX2)

		# #with tf.variable_scope("BACK_MAXPOOL1"):
		# dMP1 = tf.gradients(dX2, [relu1_out])
		# 	#print('MAXPOOL1 BACK shape =', dMP1)

		# #with tf.variable_scope("BACK_RELU1"):
		# dR1 = tf.gradients(dMP1, [conv1_out])
		# 	#print('RELU1 BACK shape =', dR1)

		# #with tf.variable_scope("BACK_CONV1"):
		# 	#print('>>> CONV5')
		# 	#dW1 = tf.Variable(tf.random_normal((11,11,3,96), mean=0, stddev=weight_scale))
		# 	#db1 = tf.Variable(tf.zeros(96))
		# dX1 = tf.gradients(dR1, X)
		# dW1 = tf.gradients(dR1, W1)
		# db1 = tf.gradients(dR1, b1)
		# 	#print('CONV1 BACK shape = ', dX1)

		conv_out = [conv1, conv2, conv3, conv4, conv5]
		relu_out = [relu1, relu2, relu3, relu4, relu5]
		max_pool_out = [maxpool1, maxpool2, maxpool3]
		affine_out = [affine1, affine2, affine3]

		params = [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6,W7,b7,W8,b8]
		# grads = [dW1,db1,dW2,db2,dW3,db3,dW4,db4,dW5,db5,dW6,db6,dW7,db7,dW8,db8]

		return loss,params,conv_out,relu_out,max_pool_out, affine_out#,grads#, dscores#, scores, data_loss, avg_data_loss, reg_loss


