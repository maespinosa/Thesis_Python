import numpy as np

from scipy.misc import imread, imresize
import os,random 
import linecache


class Imagenet2017(object): 
    def __init__(self): 
        self.foldername =''
        self.filename = ''
        self.cat_array = np.chararray((1000,1))
        self.image_size = int(227)
        
        


    def imageselect(self, save_location = ''):
        #print(sys.path)
        file = open('covered_categories.txt','r')
        match_flag = 1
        emptyfile = 0

        #print('>>> Selecting Category <<<')

        while match_flag == 1:
            match_flag = 0
            ###Randomly pick a category folder in the Training folder of the Imagenet 2017 Dataset
            dir = 'D:/Imagenet2017/ILSVRC/Data/CLS-LOC/train'
            self.foldername = random.choice(os.listdir(dir))
            #print('random category pick = ',self.foldername)
            
            ###Scan log to determine if the category has already been covered by a previous training run
            while emptyfile == 0 & match_flag == 0:
                #print('Scanning.')
                scan_file = file.readline()
                #print(scan_file)
                if scan_file == self.foldername: 
                    match_flag = 1
                    #print('category covered...reselecting')
                elif scan_file == '':
                    emptyfile = 1
                    #print('file empty')
            
            
            #print('>>> Category Selected!! <<<')
            #print('Category not previously covered in training session')

        file.close()  

        ###Write the selected category to the log
        file = open('covered_categories.txt','a')
        file.write(self.foldername + '\n')    
        file.close() 

        ###NEXT SEACH FOR A SPECIFIC FILE OF THAT CATEGORY

        ###Randomly select a file and check if its xml counter part exists
        #print('>>> Selecting Image <<<')
        xml_found = False
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0 
        rows = 0
        columns = 0
        colors = 0
        image_length = 3
        image_covered = 0
        emptyfile = 0

        while xml_found == False or xmax-xmin > self.image_size or xmax-xmin == 0 or ymax-ymin > self.image_size or ymax-ymin == 0 or rows < self.image_size or columns < self.image_size or image_length != 3 or image_covered == 1:  # or rows > self.image_size*2 or columns > self.image_size*2: 
            self.filename = random.choice(os.listdir(dir + '/' + self.foldername))

            #print(filename)
            string_length = len(self.filename)
            #print(filename[0:string_length-5])

            file = open('covered_images.txt','r')
            while (image_covered == 0 and emptyfile == 0): 
                image_file = file.readline()
                if image_file == self.filename[0:string_length-5]: 
                    image_covered = 1
                elif image_file == '': 
                    emptyfile = 1

            file.close() 



            xml_file = 'D:/Imagenet2017/ILSVRC/Annotations/CLS-LOC/train' + '/' + self.foldername +'/' + self.filename[0:string_length-5] + '.xml'
            xml_found = os.path.isfile(xml_file)

            #Find the bounding box values 
            if(xml_found == 1): 
                file = open(xml_file,'r')

                bndbox_acquired = 0
                line_number = 0

                while bndbox_acquired == 0: 
                    line_number = line_number + 1
                    #print(line_number)
                    xml_line = file.readline()
                    #print(xml_line)

                    if(line_number == 19): 
                        xmin = xml_line[9:]
                        xmin = int(float(xmin[0:len(xmin)-8]))
                    elif(line_number == 20):
                        ymin = xml_line[9:]
                        ymin = int(float(ymin[0:len(ymin)-8]))
                    elif(line_number == 21): 
                        xmax = xml_line[9:]
                        xmax = int(float(xmax[0:len(xmax)-8]))
                    elif(line_number == 22): 
                        ymax = xml_line[9:]
                        ymax = int(float(ymax[0:len(ymax)-8]))
                        bndbox_acquired = 1

                if(xmax-xmin > self.image_size or ymax-ymin > self.image_size): 
                    #print('Bounding Box larger than necessary image size of 227x227')
                    pass
                else: 
                    
                    #print('bounding box xmin = ', xmin)
                    #print('bounding box ymin = ', ymin)
                    #print('bounding box xmax = ', xmax)
                    #print('bounding box ymax = ', ymax)
                    pass

            else: 
                #print('xml not found')
                pass
            
            image = imread(dir + '/' + self.foldername + '/' + self.filename)
            rows = image.shape[0]
            columns = image.shape[1]
            image_length = len(image.shape)



            if(rows < self.image_size or columns < self.image_size): 
                #print('Image is too small')
                pass

            if(image_length != 3): 
                #print('Image Dimensions not correct')
                pass

            #print('Image dim length = ', image_length)
            #print('Image size = ', image.shape)

        #print('>>> Image Selected!! <<<')   
        #print('xml found')    
        #print(self.filename)
        #print(xml_file)

        # if(save_location != ''): 
        #     file = open(self.filename[0:len(self.filename)-5] +'.bin','wb')
        #     file.write(image)    
        #     file.close() 
        #     pass

        ###Write the selected filename to a log
        file = open('covered_images.txt','a')
        file.write(self.filename[0:string_length-5] + '\n')    
        file.close() 


        ###Determine and show the category of the image

        file = open('imagenet_categories.txt','r')
        category = ''
        category_found = 0
        category_line_counter = 0
        #category_num = 0

        while category_found == 0: 
            category_line = file.readline()

            #print(category_line)
            if category_line[0:9] == self.foldername: 
                category = category_line[10:]
                #category_num = category_line_counter
                category_found = 1
            else:
                category = ''
                category_found = 0 

            category_line_counter = category_line_counter + 1

        file.close()

        #print('>>> Imagenet Category: ', category, ' <<<')

        file = open('CLS_LOC_cat_ids.txt', 'r')
        cat_counter = 0
        id_line = ''
        id_found = 0
        class_num = 0
        file_empty = 0

        while id_found == 0 and file_empty == 0 : 

            id_line = file.readline()
            #print(id_line)
            #print(self.foldername)
            #print(len(id_line))
            #print(len(self.foldername))
            limit = min(len(id_line),len(self.foldername))
            if id_line[0:limit] == self.foldername[0:limit]:
                id_found = 1
                class_num = cat_counter#-1
            elif id_line == '': 
                file_empty = 1
            else: 
                id_found = 0
                cat_counter = cat_counter + 1
                #print(cat_counter)
        file.close()

        #print('>>> CLS LOC Class Number: ', class_num)


        if(class_num == 1000): 
            #print('Duplicate Category Detected')
            file = open('imagenet_categories.txt', 'r')
            id_line = ''
            id_found = 0
            class_desc = 0
            file_empty = 0

            while id_found == 0 and file_empty == 0 : 

                id_line = file.readline()
                #print(id_line)
                #print(self.foldername)
                #print(len(id_line))
                #print(len(self.foldername))
                limit = min(len(id_line),len(self.foldername))
                if id_line[0:limit] == self.foldername[0:limit]:
                    id_found = 1
                    class_desc = id_line[limit+1:len(id_line)]
                    #print('Class Description =', class_desc)
                elif id_line == '': 
                    file_empty = 1
                else: 
                    id_found = 0
                    
            file.close()

            file = open('CLS_LOC_cat_names.txt','r')
            id_line = ''
            id_found = 0
            file_empty = 0
            cat_counter = 0

            while id_found == 0 and file_empty == 0 : 

                id_line = file.readline()
                #print(id_line)
                #print(self.foldername)
                #print(len(id_line))
                #print(len(self.foldername))

                limit = len(id_line)
                folder_length = len(self.foldername) 
                #print(id_line[folder_length+1:limit])
                ##print(id_line[folder_length+2:limit])
                #print(id_line[folder_length+3:limit])
                #print(id_line[folder_length+4:limit])
                #print(id_line[folder_length+5:limit])
                ##print(id_line[folder_length+6:limit])
                #print(id_line[folder_length+7:limit])

                if id_line == class_desc:
                    id_found = 1
                    class_num = cat_counter#-1
                    #print('Duplicate Class ID found')
                elif id_line == '': 
                    file_empty = 1
                    #print('Duplicate Class ID NOT found')
                    #print('Class Description = ', class_desc)
                else: 
                    id_found = 0
                    cat_counter = cat_counter + 1
                    #print(cat_counter)
            file.close()




        return image, xmin, ymin, xmax, ymax, category, class_num, self.foldername, self.filename



    def imageselect_10class(self, save_location = ''):
        #print(sys.path)
        file = open('Ten_class_list.txt','r')
        match_flag = 1
        emptyfile = 0

        #Determine number of lines in file 
        for c, value in enumerate(file): 
            #print(c, value)
            num_classes = c+1;

        class_number = random.randint(1,num_classes) 
        #print('class_number = ', class_number)
        cat_desc = linecache.getline('Ten_class_list.txt',class_number)   
        #print('cat_desc = ', cat_desc)

        self.foldername = cat_desc[0:9]
        #print(self.foldername)

        class_desc = cat_desc[10:]

        file.close()
        dir = 'D:/Imagenet2017/ILSVRC/Data/CLS-LOC/train'

       ###NEXT SEACH FOR A SPECIFIC FILE OF THAT CATEGORY

        ###Randomly select a file and check if its xml counter part exists
        #print('>>> Selecting Image <<<')
        xml_found = False
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0 
        rows = 0
        columns = 0
        colors = 0
        image_length = 3
        image_covered = 0
        emptyfile = 0

        while xml_found == False or xmax-xmin > self.image_size or xmax-xmin == 0 or ymax-ymin > self.image_size or ymax-ymin == 0 or rows < self.image_size or columns < self.image_size or image_length != 3 or image_covered == 1:  # or rows > self.image_size*2 or columns > self.image_size*2: 
            self.filename = random.choice(os.listdir(dir + '/' + self.foldername))

            #print(filename)
            string_length = len(self.filename)
            #print(filename[0:string_length-5])

            file = open('covered_images.txt','r')
            while (image_covered == 0 and emptyfile == 0): 
                image_file = file.readline()
                if image_file == self.filename[0:string_length-5]: 
                    image_covered = 1
                elif image_file == '': 
                    emptyfile = 1

            file.close() 



            xml_file = 'D:/Imagenet2017/ILSVRC/Annotations/CLS-LOC/train' + '/' + self.foldername +'/' + self.filename[0:string_length-5] + '.xml'
            xml_found = os.path.isfile(xml_file)

            #Find the bounding box values 
            if(xml_found == 1): 
                file = open(xml_file,'r')

                bndbox_acquired = 0
                line_number = 0

                while bndbox_acquired == 0: 
                    line_number = line_number + 1
                    #print(line_number)
                    xml_line = file.readline()
                    #print(xml_line)

                    if(line_number == 19): 
                        xmin = xml_line[9:]
                        xmin = int(float(xmin[0:len(xmin)-8]))
                    elif(line_number == 20):
                        ymin = xml_line[9:]
                        ymin = int(float(ymin[0:len(ymin)-8]))
                    elif(line_number == 21): 
                        xmax = xml_line[9:]
                        xmax = int(float(xmax[0:len(xmax)-8]))
                    elif(line_number == 22): 
                        ymax = xml_line[9:]
                        ymax = int(float(ymax[0:len(ymax)-8]))
                        bndbox_acquired = 1

                if(xmax-xmin > self.image_size or ymax-ymin > self.image_size): 
                    #print('Bounding Box larger than necessary image size of 227x227')
                    pass
                else: 
                    
                    #print('bounding box xmin = ', xmin)
                    #print('bounding box ymin = ', ymin)
                    #print('bounding box xmax = ', xmax)
                    #print('bounding box ymax = ', ymax)
                    pass

            else: 
                #print('xml not found')
                pass
            
            image = imread(dir + '/' + self.foldername + '/' + self.filename)
            rows = image.shape[0]
            columns = image.shape[1]
            image_length = len(image.shape)



            if(rows < self.image_size or columns < self.image_size): 
                #print('Image is too small')
                pass

            if(image_length != 3): 
                #print('Image Dimensions not correct')
                pass

            #print('Image dim length = ', image_length)
            #print('Image size = ', image.shape)

        #print('>>> Image Selected!! <<<')   
        #print('xml found')    
        #print(self.filename)
        #print(xml_file)

        # if(save_location != ''): 
        #     file = open(self.filename[0:len(self.filename)-5] +'.bin','wb')
        #     file.write(image)    
        #     file.close() 
        #     pass

        ###Write the selected filename to a log
        file = open('covered_images.txt','a')
        file.write(self.filename[0:string_length-5] + '\n')    
        file.close() 



        return image, xmin, ymin, xmax, ymax, class_desc, class_number-1, self.foldername, self.filename


    def RefreshImage(self): 
        dir = 'D:/Imagenet2017/ILSVRC/Data/CLS-LOC/train'
        ref_image = imread(dir + '/' + self.foldername + '/' + self.filename)
        #print(self.foldername)
        rows,columns,colors = ref_image.shape

        #print(ref_image.shape)

        return ref_image

    def BoundingBox(self,X,xmin,ymin,xmax,ymax): 
        bbox_image = X

        bbox_image[ymin-5:ymin-1,xmin:xmax,:] = [0,255,0]
        bbox_image[ymax+1:ymax+5,xmin:xmax,:] = [0,255,0]
        bbox_image[ymin:ymax,xmin-5:xmin-1,:] = [0,255,0]
        bbox_image[ymin:ymax,xmax+1:xmax+5,:] = [0,255,0]

        return bbox_image

    def CroppedImage(self,X,xmin, ymin, xmax, ymax): 

        # Crop the image along the bounding box 

        image_cropped = X[ymin:ymax,xmin:xmax,:]
        rows, columns, colors = image_cropped.shape 
        #print(image_cropped.shape)
        #print(type(rows))

        return image_cropped, rows, columns

    def ScaleImage226(self,X,xmin, ymin, xmax, ymax, rows, columns):  

        refreshed_image = self.RefreshImage()
        #print(refreshed_image.shape)
        scaled_image = np.ones((self.image_size,self.image_size,3))
        row_delta = self.image_size-rows
        #print(row_delta)
        column_delta = self.image_size-columns
        #print(column_delta)

        row_delta_rounded = np.ceil(row_delta/2)
        column_delta_rounded = np.ceil(column_delta/2)

        if(ymin-row_delta_rounded < 0): 
            ymin = 0
            ymax = self.image_size
            #print('1')
        elif(ymax+row_delta_rounded > refreshed_image.shape[0]): 
            ymax = refreshed_image.shape[0]
            ymin = refreshed_image.shape[0]-self.image_size
            #print('2')
        else: 
            ymin = ymin-row_delta_rounded
            ymax = ymax+row_delta_rounded
            #print('5')

        if(xmin-column_delta_rounded < 0): 
            xmin = 0
            xmax = self.image_size
            #print('3')
        elif(xmax+column_delta_rounded > refreshed_image.shape[1]): 
            xmax = refreshed_image.shape[1]
            xmin = refreshed_image.shape[1]-self.image_size
            #print('4')
        else: 
            xmin = xmin-column_delta_rounded
            xmax = xmax+column_delta_rounded
            #print('6')




        temp_image = refreshed_image[int(ymin):int(ymax),int(xmin):int(xmax),:]

        scaled_image = temp_image[0:self.image_size,0:self.image_size,:]    
        scaled_rows,scaled_columns,colors = scaled_image.shape

        #print('Scaled Image size = ', scaled_image.shape)
        #print(scaled_image)

        return scaled_image

    def FileInfo(self): 
        return self.filename, self.foldername

    def CategoryNumbers(self): 

        os.remove('CLS_LOC_cat_ids.txt')
        cat_name = ''
        cat_id_found = 0
        category_line = ''
        category = ''
        emptyfile = 0

        print('Generating File')

        for i in range(1000): 
            file = open('CLS_LOC_cat_names.txt','r')
            for j in range(i+1):
                cat_name = file.readline()

            file.close()
            #print('cat name = ', cat_name)
            cat_name_length = len(cat_name)

            file = open('imagenet_categories.txt','r')
            while cat_id_found == 0 and emptyfile == 0: 
                category_line = file.readline()
                #print('category line = ', category_line)
                

                if category_line[10:10+cat_name_length] == cat_name: 
                    category = category_line[0:9]
                    cat_id_found = 1
                    #print('category found')
                elif category_line == '':
                    emptyfile = 1
                    print('file empty')


                if i == 999: 
                    #print('category_line = ', category_line[10:10+cat_name_length])
                    #print('cat name = ', cat_name)
                    #print('category_line shape = ', len(category_line[10:10+cat_name_length]))
                    #print('cat name shape = ', len(cat_name))
                    pass

            emptyfile = 0
            cat_id_found = 0
            #print('category = ', category)
            file.close()
            

            file = open('CLS_LOC_cat_ids.txt', 'a')
            file.write(category + '\n')
            file.close()



            self.cat_array[i] = category
        print('File Generated')
                
        return self.cat_array

    def get_specific_image(self, foldername = '', filename = ''): 

        dir = 'D:/Imagenet2017/ILSVRC/Data/CLS-LOC/train'
        self.foldername = foldername
        self.filename = filename
        xml_found = False
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0 
        rows = 0
        columns = 0
        colors = 0
        image_length = 3
        image_covered = 0
        emptyfile = 0
        category_id =  ''
            
        file = open('Ten_class_list.txt','r')
        id_line = ''
        id_found = 0
        file_empty = 0
        cat_counter = 0
        class_num = 0

        while id_found == 0 and file_empty == 0 : 

            id_line = file.readline()
            #print(id_line)
            #print(self.foldername)
            #print(len(id_line))
            #print(len(self.foldername))

            #limit = len(id_line)
            folder_length = len(self.foldername) 
            #print(id_line[0:folder_length])
           # print(id_line[folder_length+2:limit])
            #print(id_line[folder_length+3:limit])
            #print(id_line[folder_length+4:limit])
            #print(id_line[folder_length+5:limit])
            #print(id_line[folder_length+6:limit])
            #print(id_line[folder_length+7:limit])

            if id_line[0:folder_length] == (self.foldername):
                id_found = 1
                class_num = cat_counter#-1
                #print('Duplicate Class ID found')
            elif id_line == '': 
                file_empty = 1
                #print('Duplicate Class ID NOT found')
                #print('Class Description = ', class_desc)
            else: 
                id_found = 0
                cat_counter = cat_counter + 1
                #print(cat_counter)
        file.close()
        xml_file = 'D:/Imagenet2017/ILSVRC/Annotations/CLS-LOC/train' + '/' + self.foldername +'/' + self.filename[0:len(self.filename)-5] + '.xml'
        print(self.filename[0:len(self.filename)-5])
        xml_found = os.path.isfile(xml_file)

        #Find the bounding box values 
        if(xml_found == 1): 
            file = open(xml_file,'r')

            bndbox_acquired = 0
            line_number = 0

            while bndbox_acquired == 0: 
                line_number = line_number + 1
                #print(line_number)
                xml_line = file.readline()
                #print(xml_line)
                if(line_number == 14): 
                    category_id = xml_line[8:]
                    category_id = category_id[0:len(category_id)-8]
                    #print(category_id)
                elif(line_number == 19): 
                    xmin = xml_line[9:]
                    xmin = int(float(xmin[0:len(xmin)-8]))
                elif(line_number == 20):
                    ymin = xml_line[9:]
                    ymin = int(float(ymin[0:len(ymin)-8]))
                elif(line_number == 21): 
                    xmax = xml_line[9:]
                    xmax = int(float(xmax[0:len(xmax)-8]))
                elif(line_number == 22): 
                    ymax = xml_line[9:]
                    ymax = int(float(ymax[0:len(ymax)-8]))
                    bndbox_acquired = 1

            if(xmax-xmin > self.image_size or ymax-ymin > self.image_size): 
                print('Bounding Box larger than necessary image size of 227x227')
                

        else: 
            print('xml not found')
            #pass
        
        image = imread(dir + '/' + self.foldername + '/' + self.filename)
        print('image dtype = ', image.dtype)
        rows = image.shape[0]
        columns = image.shape[1]
        image_length = len(image.shape)



        if(rows < self.image_size or columns < self.image_size): 
            print('Image is too small')
           #pass

        if(image_length != 3): 
            print('Image Dimensions not correct')
            #pass

        return image, xmin, ymin, xmax, ymax, category_id, class_num