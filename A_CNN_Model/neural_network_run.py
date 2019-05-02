# As usual, a bit of setup
import os, random
from select_image import Imagenet2017
import sys
sys.path.append('C:/Users/Mark_Espinosa/Documents/A2/assignment2')
sys.path.append('C:/Users/Mark_Espinosa/Documents/A_CNN_Model')
#sys.path.append('C:/Users/Mark_Espinosa/AppData/Local/Programs/Python/Python37/Lib/site-packages')
#sys.path.append('C:/ProgramData/Anaconda3/Lib/site-packages')
import time
import tensorflow as tf
import shutil

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver
from alexnet import AlexNet
from alexnet_tf import AlexNet_TF
from TrainNet import TrainNet
from cs231n import optim

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

image_size = 227
std = 1e-2

Imagenet = Imagenet2017()
model = AlexNet(input_dim=(1,3,image_size,image_size), hidden_dim=4096, num_classes=1000, weight_scale=std)
gpu_model = AlexNet_TF()

Train_Net = TrainNet()


def rel_error(x, y):
	""" returns relative error """
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def clear_files():
	

	file_exists = os.path.isfile('covered_categories.txt')
	if file_exists:
	    os.remove('covered_categories.txt')

	file = open('covered_categories.txt','w')
	file.close()

	file_exists = os.path.isfile('covered_images.txt')

	if file_exists:
	    os.remove('covered_images.txt')

	file = open('covered_images.txt','w')
	file.close()

	if os.path.exists('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/input_data/'):
		shutil.rmtree('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/input_data/')

	if not os.path.exists('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/input_data/'):
		os.makedirs('C:/Users/Mark_Espinosa/Documents/A_CNN_Model/input_data/')

	#cat_array = Imagenet.CategoryNumbers()

def select_n_show(): 

	#Imagenet.CategoryNumbers()

	image,xmin,ymin,xmax,ymax,category,class_num = Imagenet.imageselect()
	plt.subplot(2,2,1)
	plt.imshow(image.astype('uint8'))
	plt.title('Original image')

	bbox_image = Imagenet.BoundingBox(image,xmin,ymin,xmax,ymax)
	cropped_image,rows,columns = Imagenet.CroppedImage(image,xmin,ymin,xmax,ymax)
	scaled_image = Imagenet.ScaleImage226(image,xmin,ymin,xmax,ymax,rows,columns)

	# Show the original image
	plt.figure(1)
	plt.subplot(2,2,2)
	plt.imshow(bbox_image.astype('uint8'))
	plt.title('Image with Bounding Box image')
	plt.subplot(2,2,3)
	plt.imshow(cropped_image.astype('uint8'))
	plt.title('Cropped image')
	plt.subplot(2,2,4)
	plt.imshow(scaled_image.astype('uint8'))
	plt.title('Scaled image')
	plt.savefig("selected_image.png")
	

	print('plotting done')

	return image, scaled_image, class_num

def alexnet_cpu(image, scaled_image, class_num): 
	image_size = 227

	std = 1e-2
	#model = AlexNet(input_dim=(3,image_size,image_size), hidden_dim=4096, num_classes=1000, weight_scale=std)
	

	print ('Testing initialization ... ')
	W1_std = abs(model.params['W1'].std() - std)
	print(model.params['W1'].std())
	#print('W1_std = ', W1_std)
	#print('W1 = ', model.params['W1'])
	b1 = model.params['b1']
	W2_std = abs(model.params['W2'].std() - std)
	b2 = model.params['b2']
	assert W1_std < std / 10, 'First layer weights do not seem right'
	assert np.all(b1 == 0), 'First layer biases do not seem right'
	assert W2_std < std / 10, 'Second layer weights do not seem right'
	assert np.all(b2 == 0), 'Second layer biases do not seem right'

	X = np.ones((1,3,image_size,image_size))
	print(X.shape)
	print(scaled_image.shape)
	X[0] = np.transpose(scaled_image, [2,0,1])


	scores = model.loss(X)
	print('scores shape = ', scores.shape)

	filename, foldername = Imagenet.FileInfo()
	print(filename)
	print(foldername)

	#y = np.asarray([foldername])
	y = np.asarray([class_num])
	loss, grads = model.loss(X, y)
	print('loss = ', loss)

	return None

def alexnet_train_gpu(num_train): 
	image_size = 227
	num_classes = 10

	std = 1e-2
	

	X = tf.placeholder(tf.float32, shape = ((num_train,3,image_size,image_size)))
	y = tf.placeholder(tf.int32, shape = ((num_train,1)))

	# X = np.ones((1,3,image_size,image_size))
	# y = np.asarray([class_num])

	#print(X.shape)
	#print(scaled_image.shape)
	#X[0] = np.transpose(scaled_image, [2,0,1])
	#print(X.shape)

	# scores = gpu_model.loss(X)
	# print('scores shape = ', scores.shape)

	filename, foldername = Imagenet.FileInfo()
	print(filename)
	print(foldername)

	#loss,params,grads = gpu_model.loss(X=X, y=y,input_dim = (num_train,3,image_size,image_size), hidden_dim = 4096, num_classes = 1000)
	loss,params,conv_out,relu_out,max_pool_out, affine_out = gpu_model.loss(X=X, y=y,input_dim = (num_train,3,image_size,image_size), hidden_dim = 4096, num_classes = num_classes)

	return X,y,loss,params,conv_out,relu_out,max_pool_out, affine_out#,grads#,scores, data_loss, avg_data_loss, reg_loss




def run_compare(): 
	
	image, scaled_image, class_num = select_n_show()
	print(scaled_image.dtype)

	start = time.time()
	cpu_loss = alexnet_cpu(image, scaled_image,class_num)
	end = time.time()
	print('CPU Elapsed Time = ', end-start)
	print('cpu_loss = ', cpu_loss)


	X,y,gpu_loss, gpu_scores, data_loss, avg_data_loss, reg_loss = alexnet_train_gpu()

	resize_scaled_image = np.ones((1,3,227,227), dtype=np.float32)
	resize_scaled_image[0] = np.transpose(scaled_image, [2,0,1])
	# resize_scaled_image = resize_scaled_image.astype(np.float32)
	print('image dtype = ', resize_scaled_image.dtype)
	y_resize = np.ones((1,1))
	y_resize[0] = class_num

	start = time.time()

	init = tf.global_variables_initializer()
	with tf.Session() as sess: 
		sess.run(init)

		feed_dict = {X: resize_scaled_image, y: y_resize}
		loss_val,X_val,y_val, score_val, dl_val, avg_loss_val, reg_loss_val = sess.run([gpu_loss,X,y,gpu_scores, data_loss, avg_data_loss, reg_loss], feed_dict)
	end = time.time()
	print('GPU Elapsed Time = ', end-start)
	print('loss = ', loss_val)
	print('Xval = ', X_val)
	print('yval = ', y_val)
	#print('scoreval = ', score_val)
	#print('data_loss_val = ', dl_val)
	print('avg_loss_val = ', avg_loss_val)
	print('reg_loss_val = ', reg_loss_val)


def run_train(): 
	

	ten_class_en = 1
	num_train = 100
	iterate_times = 10

	batches = 5#int(1000/num_train)

	# x_train, y_train = Train_Net.get_train_data(num_train=num_train)
	# print('x_train shape = ', x_train.shape)
	# print('y_train shape = ', y_train.shape)
	# #x_val, y_val = Train_Net.get_val_data(num_val=10)
	# print('y_train = ', y_train)
	X,y,gpu_loss,gpu_params,gpu_conv_out,gpu_relu_out,gpu_max_pool_out,gpu_affine_out = alexnet_train_gpu(num_train)
	#X,y,gpu_loss,params,grads = alexnet_train_gpu(num_train)
	optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(gpu_loss)
	#optimizer = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-8,False,'Adam').minimize(gpu_loss)
	# x_train = np.transpose(x_train, [0,3,1,2])
	# print('x_train shape = ', x_train.shape)
	# y_train = np.transpose(np.expand_dims(y_train, axis = 0))
	# print('y_train shape = ', y_train.shape)

	# #x_val = np.transpose(x_val, [0,3,1,2])


	W1_saver = tf.train.Saver([gpu_params[0]])
	W2_saver = tf.train.Saver([gpu_params[2]])
	W3_saver = tf.train.Saver([gpu_params[4]])
	W4_saver = tf.train.Saver([gpu_params[6]])
	W5_saver = tf.train.Saver([gpu_params[8]])
	W6_saver = tf.train.Saver([gpu_params[10]])
	W7_saver = tf.train.Saver([gpu_params[12]])
	W8_saver = tf.train.Saver([gpu_params[14]])

	b1_saver = tf.train.Saver([gpu_params[1]])
	b2_saver = tf.train.Saver([gpu_params[3]])
	b3_saver = tf.train.Saver([gpu_params[5]])
	b4_saver = tf.train.Saver([gpu_params[7]])
	b5_saver = tf.train.Saver([gpu_params[9]])
	b6_saver = tf.train.Saver([gpu_params[11]])
	b7_saver = tf.train.Saver([gpu_params[13]])
	b8_saver = tf.train.Saver([gpu_params[15]])

	conv1_saver = tf.train.Saver([gpu_conv_out[0]])
	conv2_saver = tf.train.Saver([gpu_conv_out[1]])
	conv3_saver = tf.train.Saver([gpu_conv_out[2]])
	conv4_saver = tf.train.Saver([gpu_conv_out[3]])
	conv5_saver = tf.train.Saver([gpu_conv_out[4]])

	relu1_saver = tf.train.Saver([gpu_relu_out[0]])
	relu2_saver = tf.train.Saver([gpu_relu_out[1]])
	relu3_saver = tf.train.Saver([gpu_relu_out[2]])
	relu4_saver = tf.train.Saver([gpu_relu_out[3]])
	relu5_saver = tf.train.Saver([gpu_relu_out[4]])

	maxpool1_saver = tf.train.Saver([gpu_max_pool_out[0]])
	maxpool2_saver = tf.train.Saver([gpu_max_pool_out[1]])
	maxpool3_saver = tf.train.Saver([gpu_max_pool_out[2]])

	affine1_saver = tf.train.Saver([gpu_affine_out[0]])
	affine2_saver = tf.train.Saver([gpu_affine_out[1]])
	affine3_saver = tf.train.Saver([gpu_affine_out[2]])



	init = tf.global_variables_initializer()
	with tf.Session() as sess: 
		
		sess.run(init)

		limit = batches*iterate_times

		# x_train, y_train, y_train_name = Train_Net.get_train_data(num_train=num_train)
		# print('x_train shape = ', x_train.shape)
		# print('y_train shape = ', y_train.shape)
		# #x_val, y_val = Train_Net.get_val_data(num_val=10)
		# #print('y_train = ', y_train)
		# #print('y_train_name = ', y_train_name)

		# x_train = np.transpose(x_train, [0,3,1,2])
		# #print('x_train shape = ', x_train.shape)
		# y_train = np.transpose(np.expand_dims(y_train, axis = 0))
		# #print('y_train shape = ', y_train.shape)


		#tf.write_file(filename='C:/Users/Mark_Espinosa/Documents/A_CNN_Model/W1.txt',contents='hello world')	


		for i in range(1): 
			#print('>>>>>>>>>>>>>>>> Pass %d of %d' % (i,limit))

			if(i%batches == 0): 
				clear_files()
				print('Covered Categories Deleted')
				print('Covered Images Deleted')

			save_dir = "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/input_data/"
			x_train, y_train, y_train_name = Train_Net.get_train_data(num_train=num_train, save_location=save_dir, ten_classes = ten_class_en)
			print('x_train shape = ', x_train.shape)
			print('y_train shape = ', y_train.shape)
			#x_val, y_val = Train_Net.get_val_data(num_val=10)
			#print('y_train = ', y_train)
			#print('y_train_name = ', y_train_name)

			x_train = np.transpose(x_train, [0,3,1,2])
			print('x_train shape = ', x_train.shape)
			y_train = np.transpose(np.expand_dims(y_train, axis = 0))
			print('y_train shape = ', y_train.shape)

			print('x_train value =', x_train[0,0,0:3,0:3])

			print(type(x_train))
			print(type(y_train))


			feed_dict = {X: x_train, y: y_train}

			# loss_val = sess.run([gpu_loss, optimizer], feed_dict)
			# avg_loss = np.mean(loss_val[0])
			# print('Iteration: %d of %d, loss = %f' % (i, limit, avg_loss))


			for t in range(2):
				
			# 	if(t%1 == 0):
				loss_val = sess.run([gpu_loss, optimizer], feed_dict)
				avg_loss = np.mean(loss_val[0])
			#tf.Print(loss_val, loss_val, message='loss = ')
				print('Iteration: %d of %d, loss = %f' % (t, 1000, avg_loss))
			

				if(t == 7001): 
					
					print('W1 values')
					print(sess.run(gpu_params[0][:,:,0,0]))
					print(sess.run(gpu_params[0][:,:,1,0]))
					print(sess.run(gpu_params[0][:,:,2,0]))

					#print(sess.run(gpu_params[0][0,0,0,0]))
					#print(sess.run(gpu_params[0][0,0,0,1]))
					#print(sess.run(gpu_params[0][0,0,0,2]))

					#print('')

					#print(sess.run(gpu_params[0][0,0,1,0]))
					#print(sess.run(gpu_params[0][0,0,1,1]))
					#print(sess.run(gpu_params[0][0,0,1,2]))

					#print('')

					#print(sess.run(gpu_params[0][0,1,0,0]))
					#print(sess.run(gpu_params[0][0,1,0,1]))
					#print(sess.run(gpu_params[0][0,1,0,2]))

					#print('')

					#print(sess.run(gpu_params[0][0,2,0,0]))
					#print(sess.run(gpu_params[0][0,2,0,1]))
					#print(sess.run(gpu_params[0][0,2,0,2]))

					print('')
					print('CONV1 values')
					print(sess.run(gpu_conv_out[0][0,0,:,:]))

					#print(sess.run(gpu_conv_out[0][0,0,0,0]))
					#print(sess.run(gpu_conv_out[0][0,0,0,1]))
					#print(sess.run(gpu_conv_out[0][0,0,0,2]))

					#print('')

					#print(sess.run(gpu_conv_out[0][0,0,1,0]))
					#print(sess.run(gpu_conv_out[0][0,0,1,1]))
					#print(sess.run(gpu_conv_out[0][0,0,1,2]))

					#print('')

					#print(sess.run(gpu_conv_out[0][0,1,0,0]))
					#print(sess.run(gpu_conv_out[0][0,1,0,1]))
					#print(sess.run(gpu_conv_out[0][0,1,0,2]))

					#print('')

					#print(sess.run(gpu_conv_out[0][0,2,0,0]))
					#print(sess.run(gpu_conv_out[0][0,2,0,1]))
					#print(sess.run(gpu_conv_out[0][0,2,0,2]))



		save_path = W1_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W1.ckpt")
		save_path = W2_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W2.ckpt")
		save_path = W3_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W3.ckpt")
		save_path = W4_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W4.ckpt")
		save_path = W5_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W5.ckpt")
		save_path = W6_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W6.ckpt")
		save_path = W7_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W7.ckpt")
		save_path = W8_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W8.ckpt")

		save_path = b1_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b1.ckpt")
		save_path = b2_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b2.ckpt")
		save_path = b3_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b3.ckpt")
		save_path = b4_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b4.ckpt")
		save_path = b5_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b5.ckpt")
		save_path = b6_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b6.ckpt")
		save_path = b7_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b7.ckpt")
		save_path = b8_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/b8.ckpt")


		save_path = conv1_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/conv1.ckpt")
		save_path = conv2_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/conv2.ckpt")
		save_path = conv3_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/conv3.ckpt")
		save_path = conv4_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/conv4.ckpt")
		save_path = conv5_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/conv5.ckpt")

		save_path = relu1_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/relu1.ckpt")
		save_path = relu2_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/relu2.ckpt")
		save_path = relu3_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/relu3.ckpt")
		save_path = relu4_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/relu4.ckpt")
		save_path = relu5_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/relu5.ckpt")

		save_path = maxpool1_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/maxpool1.ckpt")
		save_path = maxpool2_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/maxpool2.ckpt")
		save_path = maxpool3_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/maxpool3.ckpt")

		save_path = affine1_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/affine1.ckpt")
		save_path = affine2_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/affine2.ckpt")
		save_path = affine3_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/layer_output/affine3.ckpt")

					#save_path = image_saver.save(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/input_data/input_image_data.ckpt")

	sess.close()		


def run_test(): 
	
	
	#W1 = tf.get_variable("W1", [11,11,3,96], initializer = tf.zeros_initializer)
	#saver = tf.train.Saver({"W1": W1})
	X,gpu_loss,params,conv_out,relu_out,max_pool_out, affine_out = alexnet_test_gpu()
	#W1_saver = tf.train.Saver([params[0]])

	start = time.time()

	init = tf.global_variables_initializer()
	with tf.Session() as sess: 
		sess.run(init)
		#W1.initializer.run()
		#W1_saver.restore(sess, "C:/Users/Mark_Espinosa/Documents/A_CNN_Model/params/W1.ckpt")#.data-00000-of-00001')

		#image_data = get_single_image(self, file_name = 'n01614925', folder_name = '2438')
		#print('image_data shape = ', image_data.shape)
		image_data,y = Train_Net.get_single_image(file_name = 'n01614925_2438.jpeg', folder_name = 'n01614925')
		print('image_data shape = ', image_data.shape)
		#x_test = np.transpose(image_data, [2,0,1])
		#print(x_test.dtype)
		image_data = np.transpose(image_data, [0,3,1,2])
		#x_reshape = np.zeros([1,x_test.shape[0],x_test.shape[1],x_test.shape[2]]) 
		#x_reshape[0,:,:,:] = x_test
		#image_data= tf.cast(image_data, tf.float32)
		print(image_data.dtype)

		# y = np.zeros([2,1])
		# print(y.dtype)
		# print('y shape ',y.shape)
		# y[0,0] = 5.0

		#print(type(image_data))
		#print(type(y))


		feed_dict = {X: image_data}#, y: y}
		loss_val = sess.run([gpu_loss], feed_dict)
	end = time.time()
	print('GPU Elapsed Time = ', end-start)
	print('loss = ', loss_val)

	sess.close()		



def alexnet_test_gpu(): 
	image_size = 227
	num_classes = 10

	std = 1e-2

	X = tf.placeholder(dtype = tf.float32, shape = ((1,3,image_size,image_size)))
	#y = tf.placeholder(dtype = tf.int32, shape = ((5,1)))



	filename, foldername = Imagenet.FileInfo()
	print(filename)
	print(foldername)

	#loss,params,grads = gpu_model.loss(X=X, y=y,input_dim = (num_train,3,image_size,image_size), hidden_dim = 4096, num_classes = 1000)
	loss,params,conv_out,relu_out,max_pool_out, affine_out = gpu_model.loss(X=X, y=None,input_dim = (1,3,image_size,image_size), hidden_dim = 4096, num_classes = num_classes, get_saved_params = 1)

	return X,loss,params,conv_out,relu_out,max_pool_out, affine_out#,grads#,scores, data_loss, avg_data_loss, reg_loss







if __name__ == '__main__':
	tf.Session().close()
	#Imagenet.CategoryNumbers()
	#clear_files()
	#run_compare()
	#print('DEBUG HERE')
	run_test()
	#run_train()
