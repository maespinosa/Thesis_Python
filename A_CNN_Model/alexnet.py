import numpy as np

import sys
sys.path.append('C:/Users/Mark_Espinosa/Documents/A2/assignment2')

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast
from time import time
from cs231n.im2col import *
import struct

import os

class AlexNet(object):
	def __init__(self, input_dim=(3, 224, 224), hidden_dim=4096, num_classes=10, weight_scale=1e-3, reg=0.0, #reg_decay = 1.0,
               dtype=np.float32):

		self.params = {}
		self.reg = reg
		self.dtype = dtype
		self.image_size = 227
		#self.reg_decay = reg_decay
		#self.outputs = {}
		
		W1 = np.random.normal(0,weight_scale,96*11*11*3)
		W1 = np.reshape(W1, [96,3,11,11])
		b1 = np.zeros([96])
					
		W2 = np.random.normal(0,weight_scale,256*5*5*96)
		W2 = np.reshape(W2, [256,96,5,5])
		b2 = np.zeros([256])
			
		W3 = np.random.normal(0,weight_scale,384*3*3*256)
		W3 = np.reshape(W3, [384,256,3,3])
		b3 = np.zeros([384])	
			
		W4 = np.random.normal(0,weight_scale,384*3*3*384)
		W4 = np.reshape(W4, [384,384,3,3])
		b4 = np.zeros([384])
			
		W5 = np.random.normal(0,weight_scale,256*3*3*384)
		W5 = np.reshape(W5, [256,384,3,3])
		b5 = np.zeros([256])
			
		W6 = np.random.normal(0,weight_scale,hidden_dim*6*6*256)
		W6 = np.reshape(W6, [256*6*6,hidden_dim])
		b6 = np.zeros([hidden_dim])	
			
		W7 = np.random.normal(0,weight_scale,hidden_dim*hidden_dim)
		W7 = np.reshape(W7, [hidden_dim,hidden_dim])
		b7 = np.zeros([hidden_dim])
			
		W8 = np.random.normal(0,weight_scale,hidden_dim*num_classes)
		W8 = np.reshape(W8, [hidden_dim,num_classes])
		b8 = np.zeros([num_classes])
			

		self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3, 'W4': W4, 'b4':b4, 'W5':W5, 'b5': b5, 'W6': W6, 'b6': b6, 'W7': W7,'b7': b7, 'W8': W8,'b8': b8}

	def get_saved_parameters(self, hidden_dim = 4096, num_classes = 10): 
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

		    self.params['W1'] = np.reshape(data, [96,3,11,11])

		
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
		
		
		# ## b6 Init ---------------------------------------------------
		# file_exists = os.path.isfile('./params_temp/b6.bin')
		# file_size = os.path.getsize('./params_temp/b6.bin')

		# data = np.zeros([int(file_size/8)])
		# if file_exists:
		#     print('b6.bin FOUND!!')
		#     file = open('./params_temp/b6.bin','rb')
		#     for i in range(int(file_size/8)):
		#         data[i] = struct.unpack('d', file.read(8))[0]
		#     file.close()

		#     self.params['b6'] = np.reshape(data, [hidden_dim])
			
			
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
		
		# ## b7 Init ---------------------------------------------------
		# file_exists = os.path.isfile('./params_temp/b7.bin')
		# file_size = os.path.getsize('./params_temp/b7.bin')

		# data = np.zeros([int(file_size/8)])
		# if file_exists:
		#     print('b7.bin FOUND!!')
		#     file = open('./params_temp/b7.bin','rb')
		#     for i in range(int(file_size/8)):
		#         data[i] = struct.unpack('d', file.read(8))[0]
		#     file.close()

		#     self.params['b7'] = np.reshape(data, [hidden_dim])
			
			
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
		
		
		# ## b8 Init ---------------------------------------------------
		# file_exists = os.path.isfile('./params_temp/b8.bin')
		# file_size = os.path.getsize('./params_temp/b8.bin')

		# data = np.zeros([int(file_size/8)])
		# if file_exists:
		#     print('b8.bin FOUND!!')
		#     file = open('./params_temp/b8.bin','rb')
		#     for i in range(int(file_size/8)):
		#         data[i] = struct.unpack('d', file.read(8))[0]
		#     file.close()

		#     self.params['b8'] = np.reshape(data, [num_classes])
			
			


	def loss(self, X, y=None,save_en=0, reg = 0.0):
		"""
		Evaluate loss and gradient for the three-layer convolutional network.

		Input / output: Same API as TwoLayerNet in fc_net.py.
		"""
		self.reg = reg
		#print('reg = ', self.reg)
		

		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		W3, b3 = self.params['W3'], self.params['b3']
		W4, b4 = self.params['W4'], self.params['b4']
		W5, b5 = self.params['W5'], self.params['b5']
		W6, b6 = self.params['W6'], self.params['b6']
		W7, b7 = self.params['W7'], self.params['b7']
		W8, b8 = self.params['W8'], self.params['b8']
		
		
		#get_saved_parameters()


		#print('W1 shape = ', W1.shape)

		# pass conv_param to the forward pass for the convolutional layer
		filter_size = W1.shape[2]
		conv1_param = {'stride': 4, 'pad': 0}
		conv2_param = {'stride': 1, 'pad': 2}
		conv3_param = {'stride': 1, 'pad': 1}
		conv4_param = {'stride': 1, 'pad': 1}
		conv5_param = {'stride': 1, 'pad': 1}				

		# pass pool_param to the forward pass for the max-pooling layer
		pool_param = {'pool_height': int(3), 'pool_width': int(3), 'stride': int(2)}

		scores = None

		pass
		#print(X[0,0,0,:])
		#print(X[0,1,0,:])
		#print(X[0,2,0,:])
		#print('X shape =', X.shape)
		#print('x dtype = ', X.dtype)
		row_mean = np.mean(X)
		print('Mean = ',row_mean)
		#print('Mean shape= ',row_mean.shape)

		X1 = X - row_mean
		#print("X1 = ", X1)
		std_dev = np.std(X1)
		print('std_dev = ',std_dev)
		X2 = X1 / std_dev
		#print("Norm = ", X2[0,0,0,:])
		#X1 = X# np.subtract(X,128)



		#print(X[0,0:3,0:3])
		#print(X1[0,0:3,0:3])
		#print('means = ',np.mean(X,axis = 0))

		  #conv - relu - 2x2 max pool - affine - relu - affine - softmax
		#print('>>> CONV1')
		#conv1_out, conv1_cache = conv_forward_naive(X, W1, b1, conv1_param)
		conv1_out, conv1_cache = conv_forward_fast(X2, W1, b1, conv1_param)
		#print('conv1_out shape = ', conv1_out.shape)
		#print('conv1_out = ', conv1_out[0,0,0,0:55])
		#print('conv1_out = ', conv1_out[0,0,1,0:11])
		#print('conv1_out = ', conv1_out[0,1,0,0:11])
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/input_image.bin','wb')
			file.write(X)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W1.bin','wb')
			file.write(W1)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b1.bin','wb')
			file.write(b1)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/conv1.bin','wb')
			file.write(conv1_out)
		#print('conv1_out size = ', conv1_out.shape)
		relu1_out, relu1_cache = relu_forward(conv1_out)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/relu1.bin','wb')
			file.write(relu1_out)
		#print('relu1_out size = ', relu1_out.shape)
		#print('relu1_out = ', relu1_out[0,0,0,0:11])
		#print('relu1_out = ', relu1_out[0,0,1,0:11])
		#print('relu1_out = ', relu1_out[0,1,0,0:11])


		#print('>>> MAXPOOL1')
		#max_pool1_out, pool1_cache = max_pool_forward_naive(relu1_out, pool_param)
		max_pool1_out, pool1_cache = max_pool_forward_fast(relu1_out, pool_param)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/maxpool1.bin','wb')
			max_pool1_out = max_pool1_out.copy(order='C')
			file.write(max_pool1_out)
		#print('max_pool1_out size = ', max_pool1_out.shape)
		#print('max_pool1_out = ', max_pool1_out[0,0,0,0:11])
		#print('max_pool1_out = ', max_pool1_out[0,0,1,0:11])
		#print('max_pool1_out = ', max_pool1_out[0,1,0,0:11])

		#print('>>> CONV2')
		conv2_out, conv2_cache = conv_forward_fast(max_pool1_out, W2, b2, conv2_param)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W2.bin','wb')
			file.write(W2)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b2.bin','wb')
			file.write(b2)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/conv2.bin','wb')
			file.write(conv2_out)
		
		relu2_out, relu2_cache = relu_forward(conv2_out)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/relu2.bin','wb')
			file.write(relu2_out)
		#print('>>> MAXPOOL2')
		max_pool2_out, pool2_cache = max_pool_forward_fast(relu2_out, pool_param)
		if(save_en == 1): 	
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/maxpool2.bin','wb')
			max_pool2_out = max_pool2_out.copy(order='C')
			file.write(max_pool2_out)
		#print('>>> CONV3')
		conv3_out, conv3_cache = conv_forward_fast(max_pool2_out, W3, b3, conv3_param)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W3.bin','wb')
			file.write(W3)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b3.bin','wb')
			file.write(b3)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/conv3.bin','wb')
			file.write(conv3_out)

		relu3_out, relu3_cache = relu_forward(conv3_out)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/relu3.bin','wb')
			file.write(relu3_out)
		#print('>>> CONV4')
		conv4_out, conv4_cache = conv_forward_fast(relu3_out, W4, b4, conv4_param)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W4.bin','wb')
			file.write(W4)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b4.bin','wb')
			file.write(b4)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/conv4.bin','wb')
			file.write(conv4_out)
		#print(conv4_out[0,0,: ,:])
		relu4_out, relu4_cache = relu_forward(conv4_out)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/relu4.bin','wb')
			file.write(relu4_out)
		#print('>>> CONV5')
		conv5_out, conv5_cache = conv_forward_fast(relu4_out, W5, b5, conv5_param)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W5.bin','wb')
			file.write(W5)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b5.bin','wb')
			file.write(b5)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/conv5.bin','wb')
			file.write(conv5_out)
		
		relu5_out, relu5_cache = relu_forward(conv5_out)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/relu5.bin','wb')
			file.write(relu5_out)
		#print('>>> MAXPOOL3')
		max_pool3_out, pool3_cache = max_pool_forward_fast(relu5_out, pool_param)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/maxpool3.bin','wb')
			max_pool3_out = max_pool3_out.copy(order='C')
			file.write(max_pool3_out)
		#print('>>> Affine1')
		affine1_out, affine1_cache = affine_forward(max_pool3_out, W6, b6)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W6.bin','wb')
			file.write(W6)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b6.bin','wb')
			file.write(b6)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/affine1.bin','wb')
			file.write(affine1_out)
		#print('affine1_out shape = ', affine1_out.shape)
		#print('affine1_out = ', affine1_out[0,0:10])
		#print('>>> Affine2')
		affine2_out, affine2_cache = affine_forward(affine1_out, W7, b7)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W7.bin','wb')
			file.write(W7)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b7.bin','wb')
			file.write(b7)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/affine2.bin','wb')
			file.write(affine2_out)
		#print('affine2_out shape = ', affine2_out.shape)
		#print('affine2_out = ', affine2_out[0,0:10])	
		#print('>>> Affine3')
		affine3_out, affine3_cache = affine_forward(affine2_out, W8, b8)
		if(save_en == 1): 
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W8.bin','wb')
			file.write(W8)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b8.bin','wb')
			file.write(b8)
			file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/affine3.bin','wb')
			file.write(affine3_out)
		#print('>>> Scores')
		scores = affine3_out

		print('scores shape = ', scores.shape)


		self.outputs = {'CONV1': conv1_out, 'RELU1': relu1_out, 'MAXPOOL1': max_pool1_out, 
					'CONV2': conv2_out, 'RELU2': relu2_out, 'MAXPOOL2': max_pool2_out, 
					'CONV3': conv3_out, 'RELU3': relu3_out, 'MAXPOOL3': max_pool3_out, 
					'CONV4': conv4_out, 'RELU4': relu4_out, 
					'CONV5': conv5_out, 'RELU5': relu5_out,
					'AFFINE1': affine1_out, 'AFFINE2': affine2_out, 'AFFINE3': affine3_out}

		#print('AlexNet Forward Pass Done')


		if y is None:
		  return scores

		loss, grads = 0, {}

		pass

		#print("SOFTMAX ==================")
		#print('Y = ', y)
		#print('Y shape = ', y.shape)


		softmax_input = scores; 
		#print('softmax_input shape is = ', softmax_input.shape)	

		softmax_input_trans = np.transpose(softmax_input)
		#print('softmax_input_trans shape = ', softmax_input_trans.shape)

		exp_scores = np.exp(softmax_input_trans)
		#print('exp of scores = ', exp_scores)

		sum_exp_scores = np.sum(exp_scores, axis = 0)
		#print('sum of exp scores = ', sum_exp_scores)


		P = np.empty([softmax_input_trans.shape[0], softmax_input_trans.shape[1]])
		Py = np.empty(y.shape)
		#print(y.shape)
		for i in range(y.shape[0]): 
		    #print(i)
		    P[:,i] = exp_scores[:,i] / sum_exp_scores[i]

		file =  open('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/softmax.bin','wb')
		file.write(P)
		#print('P = ',P)
		#print('P shape = ', P.shape)

		#print("Probabilities = ", P)
		for i in range(int(y.shape[0])):
			#print(y[i])
			Py[i] = P[int(y[i]),i]
		#print('Py = ', Py)	

		data_loss = -1*np.log(Py)

		#print('data loss = ', data_loss)
		#print('reg = ', self.reg)

		avg_data_loss =  (1/softmax_input_trans.shape[1]) * np.sum(data_loss)
		# print('average data loss = ', avg_data_loss)
		#print('average data loss shape = ', avg_data_loss.shape)

		reg_loss = 0.5*self.reg*(np.sum(W1**2) + np.sum(W2**2)  + np.sum(W3**2) + np.sum(W4**2) + np.sum(W5**2) + np.sum(W6**2) + np.sum(W7**2) + np.sum(W8**2))
		#print('regularization loss = ', reg_loss)
		#print('regularization loss = ', reg_loss.shape)

		loss = avg_data_loss + reg_loss

		#print('loss = ', loss)

		# compute the gradient on scores
		dscores = P
		#print('P = ', P)
		#print('P shape = ', P.shape)
		for i in range(int(y.shape[0])):
		     dscores[int(y[i]),i]  = dscores[int(y[i]),i] - 1.0
		#print('dscores = ', dscores)

		dscores /= X.shape[0]
		#print('dscores = ', dscores)


		#print('Backprop Affine3')
		dx8,dW8,db8 = affine_backward(np.transpose(dscores), affine3_cache)
		#print('dx8 shape = ', dx8.shape)
		#print('dW8 shape = ', dW8.shape)
		#print('Backprop Affine2')
		dx7,dW7,db7 = affine_backward(dx8, affine2_cache)
		#print('dx7 shape = ', dx7.shape)
		#print('dW7 shape = ', dW7.shape)
		#print('Backprop Affine1')
		dx6,dW6,db6 = affine_backward(dx7, affine1_cache)
		#print('dx6 shape = ', dx6.shape)
		#print('dW6 shape = ', dW6.shape)
		#print('Backprop MaxPool3')
		dx3_max = max_pool_backward_fast(dx6, pool3_cache)
		#print('dx3_max shape = ', dx3_max.shape)
		#print('Backprop Relu5')
		dx5_relu = relu_backward(dx3_max, relu5_cache)
		#print('dx5_relu shape = ', dx5_relu.shape)
		#print('Backprop Conv5')
		dx5,dW5,db5 = conv_backward_fast(dx5_relu, conv5_cache)
		#print('dx5 shape = ', dx5.shape)
		#print('dW5 shape = ', dW5.shape)
		#print('Backprop Relu4')
		dx4_relu = relu_backward(dx5, relu4_cache)
		#print('dx4_relu shape = ', dx4_relu.shape)
		#print('Backprop Conv4')
		dx4,dW4,db4 = conv_backward_fast(dx4_relu, conv4_cache)
		#print('dx4 shape = ', dx4.shape)
		#print('dW4 shape = ', dW4.shape)
		#print('Backprop Relu3')
		dx3_relu = relu_backward(dx4, relu3_cache)
		#print('dx3_relu shape = ', dx3_relu.shape)
		#print('Backprop Conv3')
		dx3,dW3,db3 = conv_backward_fast(dx3_relu, conv3_cache)
		#print('dx3 shape = ', dx3.shape)
		#print('dW3 shape = ', dW3.shape)
		#print('Backprop MaxPool2')
		dx2_max = max_pool_backward_fast(dx3, pool2_cache)
		#print('dx2_max shape = ', dx2_max.shape)
		#print('Backprop Relu2')
		dx2_relu = relu_backward(dx2_max, relu2_cache)
		#print('dx2_relu shape = ', dx2_relu.shape)
		#print('Backprop Conv2')
		dx2,dW2,db2 = conv_backward_fast(dx2_relu, conv2_cache)
		#print('dx2 shape = ', dx2.shape)
		#print('dW2 shape = ', dW2.shape)
		#print('Backprop MaxPool1')
		dx1_max = max_pool_backward_fast(dx2, pool1_cache)
		#print('dx1_max shape = ', dx1_max.shape)
		#print('Backprop Relu1')
		dx1_relu = relu_backward(dx1_max, relu1_cache)
		#print('dx1_relu shape = ', dx1_relu.shape)
		#print('Backprop Conv1')
		dx1,dW1,db1 = conv_backward_fast(dx1_relu, conv1_cache)
		#print('dx1 shape = ', dx1.shape)
		#print('dW1 shape = ', dW1.shape)

		#print('AlexNet Backward Pass Done')

		#self.reg *= self.reg_decay
		
		dW1 += self.reg*W1
		dW2 += self.reg*W2
		dW3 += self.reg*W3
		dW4 += self.reg*W4
		dW5 += self.reg*W5
		dW6 += self.reg*W6
		dW7 += self.reg*W7
		dW8 += self.reg*W8

		grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5, 'W6':dW6, 'b6':db6, 'W7':dW7, 'b7':db7, 'W8':dW8, 'b8':db8}

		return loss, grads, self.outputs

