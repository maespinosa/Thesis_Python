import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import sys
sys.path.append('C:/Users/Marks-M3800/Documents/A2/assignment2')

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


		cat_array = self.Imagenet.CategoryNumbers()

	def get_train_data(self, num_train=100,save_location=''): 
		x_train = np.zeros([num_train,self.image_size,self.image_size,3])
		y_train = np.zeros([num_train])
		y_train_name = np.chararray([num_train],itemsize=9)



		for i in range(num_train):
			#print('Training Image: ', i)

			image,xmin,ymin,xmax,ymax,category,class_num, class_name, filename = self.Imagenet.imageselect(save_location)
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
			y_train[i] = class_num
			y_train_name[i] = class_name

			if(save_location != ''): 
				scipy.misc.imsave('./input_data/' + filename[0:len(filename)-5] + '_image.jpg', image)
				scipy.misc.imsave('./input_data/' + filename[0:len(filename)-5] + '_cropped.jpg', cropped_image)
				scipy.misc.imsave('./input_data/' + filename[0:len(filename)-5] + '_scaled.jpg', scaled_image)
				pass

		return x_train, y_train, y_train_name

	def get_val_data(self, num_val=100):
		x_val = np.zeros([num_val,self.image_size,self.image_size,3])
		y_val = np.zeros([num_val,1])

		for i in range(num_val):
			#print('Validation Image: ', i)

			image,xmin,ymin,xmax,ymax,category,class_num,filename = self.ValData.image_select()
			print('class_num = ', class_num)
			print('category = ', category)
			#print('filename = ', filename)	

			bbox_image = self.ValData.BoundingBox(image,xmin,ymin,xmax,ymax)
			cropped_image,rows,columns = self.ValData.CroppedImage(image,xmin,ymin,xmax,ymax)
			scaled_image = self.ValData.ScaleImage226(image,xmin,ymin,xmax,ymax,rows,columns)

			x_val[i,:,:,:] =  scaled_image
			y_val[i] = class_num

		return x_val, y_val




