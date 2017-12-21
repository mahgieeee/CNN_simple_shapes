"""
Created on Sun Nov 12 18:41:03 2017
Program finds contours by using the sobel algorithm. After finding
the countours, rectangular boundaries are drawn around it. The 
locations of the rectangles are in (x,y) coordinates where the cropped
version will be from the top left corner (smallest_x, smallest_y)
and the bottom right opposite corner (largest_x, largest y). get_edges
creates a copy of the image that needs to be in grayscale for the 
sobel algorithm. After the boundary is found, cropping of the original
image will be done through array slicing. This python 2 program will be 
implemented in the main cnn loader file, which will be incorporated with 
the cropping of images as a function. The cnn will see the images cropped 
and will not see the original image.  
Nov 16: If offset is 50 or greater, the offset extends the original image when 
pasted on the background (ValueError: tile cannot extend outside image)
Need to convert np.array16 to np.array32 since cv2 doesn't support np.array16.
This program saves the cropped images of 200x200 in a folder.
@author: maggie
"""
from __future__ import print_function
import cv2
import numpy as np
from tensorflow.python.lib.io import file_io 
import joblib
import argparse
from PIL import Image
import scipy.misc

"""to run code locally:
   python cnn_sobel_main.py --save-pathname test_resized_images/circle/circle_ --train-file circles_background_png.pkl --num-images 2000
"""
     
"""code to get boundaries of contour shapes and crops the images based on the 
location of the rectangular boundaries"""
def get_edges(image_array, background_color):    
    # needed for cv2 to read the image in the proper color format
    new_image_converted = image_array.astype(np.float32)
    #since drawContours modifies the image make a copy of the input 
    new_image_copy = new_image_converted.copy() 
    original_img = cv2.cvtColor(new_image_copy, cv2.COLOR_BGR2RGB)

    # sobel algorithm requires the image to be in grayscale
    grayscale = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # gaussian to remove noise from 
    img = cv2.GaussianBlur(grayscale,(3,3),0)
    # use sobel algorithm to detect contours
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  
    gradient_x = cv2.convertScaleAbs(sobelx)
    gradient_y = cv2.convertScaleAbs(sobely)
    
    try:
        # combine two sobel gradients
        gradient_t = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
        contours = []
        im2, contours, hierarchy = cv2.findContours(gradient_t, 
                                                    cv2.RETR_TREE, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
        # with each contour, find 2 sets of coordinates to draw bounding rectangle
        edge_list_x = []
        edge_list_y = []	
        for c in contours:
            x, y, width, height = cv2.boundingRect(c)
            # draw bounding rectangles around contours
            cv2.rectangle(grayscale, (x,y), (x+width, y+height), (0,255,0), 2) 
            edge_list_x.append(x)
            edge_list_y.append(y)
            edge_list_x.append(x+width)
            edge_list_y.append(y+height)
        smallest_x = min(edge_list_x)
        smallest_y = min (edge_list_y)
        largest_x = max(edge_list_x)
        largest_y = max(edge_list_y)
        
        crop_image = new_image_copy[smallest_y:largest_y, 
                                    smallest_x:largest_x]
        crop_image = np.array(crop_image, np.float32)
        width, height = crop_image.shape[0], crop_image.shape[1]
        print (width, height)
        (r,g,b) = background_color
        print ((r, g, b))
        rgb_color = (int(r*255), int(g*255), int(b*255))
    
        if width < 200 and height < 200:
        		offset = 5
        		crop_original_image = image_array[smallest_y - offset:largest_y + offset, 
        										  smallest_x - offset:largest_x + offset]	  
        		# same color background to create the same shapes (needed for keras generator)
        		resize_width, resize_height = crop_original_image.shape[0], crop_original_image.shape[1]
        		pasted_crop = Image.new("RGB", (200, 200), color = rgb_color)
        		# creating an image from a PIL numpy array
        		new_image = Image.fromarray(np.uint8(crop_original_image))
        		pasted_crop.paste(new_image, (0, 0, resize_height, resize_width))
        		# convert back to numpy array
        		pasted_crop = np.array(pasted_crop, np.float32)
        		return pasted_crop
    		
    		
        if width >= 200 or height >= 200:                 
        		offset = 5
        		crop_original_image = image_array[smallest_y:largest_y + offset, 
                                                  smallest_x:largest_x + offset]
        		#crop_original_image = np.array(crop_original_image, np.float16)
            
        		#resizes the image with a fixed width of 300 with respect to its aspect ratio 
        		crop_image = Image.fromarray(np.uint8(crop_original_image))
        		crop_width = 200
        		percent = (crop_width / float(crop_image.size[0]))
        		crop_height = int((float(crop_image.size[1]) * float(percent)))
        		crop_image = crop_image.resize((crop_width, crop_height), Image.ANTIALIAS) #ANTIALIAS reserves quality
        		crop_image = np.array(crop_image, np.float32)
        		resize_width, resize_height = crop_image.shape[0], crop_image.shape[1]
        		pasted_crop = Image.new("RGB", (200, 200), color = rgb_color)
        		# creating an image from a PIL numpy array
        		crop_image = Image.fromarray(np.uint8(crop_image))
        		pasted_crop.paste(crop_image, (0, 0, resize_height, resize_width))
        		# convert back to numpy array
        		pasted_crop = np.array(pasted_crop, np.float32)
        		return pasted_crop
    except:
        return image_array

def train_model(num_images = '200',
                train_file = 'circles_background_png.pkl',
                save_pathname = 'test_resized_images/circle/circle_', 
                **args):
    with file_io.FileIO(train_file , mode='r') as f:
        # joblib loads compressed files consistenting of large datasets 
        save = joblib.load(f)
        circles = save['circles']
        background_color = save['train_y_dataset']
        del save  # hint to help gc free up memory 
        
    num = int(num_images)  
    circle_name = []
    for i in range(num):
        circle_name.append(save_pathname + str(i) + ".png")
    for x in range(num):
        scipy.misc.imsave(circle_name[x], get_edges(circles[x], background_color[x]))

        
if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-images',
                        help='number of images to be saved')
    parser.add_argument('--train-file',
                        help='local path of pickle file')
    parser.add_argument('--save-pathname', 
                        help='pathname of file images to be saved after being cropped')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)

    