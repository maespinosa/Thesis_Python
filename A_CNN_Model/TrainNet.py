import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import sys
import os,random 
sys.path.append('C:/Users/Mark_Espinosa/Documents/A2/assignment2')

from select_image import Imagenet2017
from validation_data import ValidationData

class TrainNet(object):
	def __init__(self):
		self.image_size = 227
		self.Imagenet = Imagenet2017()
		self.ValData = ValidationData() 

		pass 
	def clean_all(self): 
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

		file_exists = os.path.isfile('validation_covered_images.txt')

		if file_exists:
		    os.remove('validation_covered_images.txt')

		file = open('validation_covered_images.txt','w')
		file.close()


		cat_array = self.Imagenet.CategoryNumbers()

	def get_train_data(self, num_train=100,save_location='',ten_classes=0,clean_files = 0): 
		x_train = np.zeros([num_train,self.image_size,self.image_size,3])
		y_train = np.zeros([num_train])
		y_train_name = np.chararray([num_train],itemsize=9)

		if(clean_files == 1): 
			self.clean_all()



		for i in range(num_train):
			print('Training Image: ', i)
			if(ten_classes == 0): 
				image,xmin,ymin,xmax,ymax,category,class_num, class_name, filename = self.Imagenet.imageselect(save_location)
			else:
				image,xmin,ymin,xmax,ymax,category,class_num, class_name, filename = self.Imagenet.imageselect_10class(save_location)

			#print('class_num = ', class_num)
			#print('class_name = ', class_name)
			#print('filename = ', filename)			
			#print('Selected Image Shape :',image.shape)
			#bbox_image = self.Imagenet.BoundingBox(image,xmin,ymin,xmax,ymax)
			#print('Image With Bounding Box Shape: ', bbox_image.shape)
			cropped_image,rows,columns = self.Imagenet.CroppedImage(image,xmin,ymin,xmax,ymax)
			#print('Cropped Image Shape: ', cropped_image.shape)
			scaled_image = self.Imagenet.ScaleImage226(image,int(xmin),int(ymin),int(xmax),int(ymax),int(rows),int(columns))
			#print('Scaled Image Shape: ', scaled_image.shape)
			

			x_train[i,:,:,:] =  scaled_image
			y_train[i] = int(class_num)
			y_train_name[i] = class_name

			if(save_location != ''): 
				scipy.misc.imsave('./input_data/' + filename[0:len(filename)-5] + '_image.jpg', image)
				scipy.misc.imsave('./input_data/' + filename[0:len(filename)-5] + '_cropped.jpg', cropped_image)
				scipy.misc.imsave('./input_data/' + filename[0:len(filename)-5] + '_scaled.jpg', scaled_image)
				pass

		return x_train, y_train, y_train_name

	def get_val_data(self, num_val=100, ten_classes = 0):
		x_val = np.zeros([num_val,self.image_size,self.image_size,3])
		y_val = np.zeros([num_val])
		incount = 0

		for i in range(num_val):
			print('Validation Image: ', i)
			if(ten_classes == 0):
				image,xmin,ymin,xmax,ymax,category,class_num,filename = self.ValData.image_select()
			else:
				image,xmin,ymin,xmax,ymax,category,class_num,filename,out_count = self.ValData.image_select_10class(incount = incount)


			#print('class_num = ', class_num)
			#print('category = ', category)
			#print('filename = ', filename)	
			incount = out_count

			bbox_image = self.ValData.BoundingBox(image,xmin,ymin,xmax,ymax)
			cropped_image,rows,columns = self.ValData.CroppedImage(image,xmin,ymin,xmax,ymax)
			scaled_image = self.ValData.ScaleImage226(image,xmin,ymin,xmax,ymax,rows,columns)

			x_val[i,:,:,:] =  scaled_image
			y_val[i] = int(class_num)


		return x_val, y_val


	def get_single_image(self, file_name = '', folder_name = ''):
		x_single = np.zeros([1,self.image_size,self.image_size,3])
		y_single = np.zeros([1,1])
		image, xmin, ymin, xmax, ymax, category_id, y = self.Imagenet.get_specific_image(foldername = folder_name, filename = file_name)

		#print('class_num = ', class_num)
		#print('category = ', category)
		#print('filename = ', filename)	
		print('y = ', y)

		bbox_image = self.Imagenet.BoundingBox(image,xmin,ymin,xmax,ymax)
		cropped_image,rows,columns = self.Imagenet.CroppedImage(image,xmin,ymin,xmax,ymax)
		scaled_image = self.Imagenet.ScaleImage226(image,xmin,ymin,xmax,ymax,rows,columns)
		print('scaled_image dtype = ', scaled_image.dtype)
		print('scaled image shpae = ', scaled_image.shape)
		print('red = ', scaled_image[0,:,0])
		x_single[:,:,:,:] =  scaled_image
		y_single[:,0] = y 

		return x_single, y_single

