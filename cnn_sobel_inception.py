"""
Program uses the inception module architecture to see if cnn will improve  
@author: maggie
"""
from __future__ import print_function
from __future__ import division
import argparse
import cv2
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import concatenate
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
#from sklearn.metrics import f1_score
from datetime import datetime 
from tensorflow.python.lib.io import file_io 
from PIL import Image
import h5py
import joblib
import numpy as np

"""to run code locally:
   python cnn_sobel_lstm.py --job-dir ./ --train-file random_shapes_cropped.pkl
"""

"""code to get boundaries of contour shapes and crops the images based on the 
location of the rectangular boundaries. The cropped image has to be within the 
same shape as the original image (300, 300, 3) - pasted the cropped image."""
def get_edges(image_array):    
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
    # combine two sobel gradients
    gradient_t = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
    contours = []
    im2, contours, hierarchy = cv2.findContours(gradient_t, 
                                                cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_SIMPLE)
    # generator function returns empty contours [] list, stopping the 
    # image generator, it will return the original array generator gives 
    # empty contours
    if not contours:
        return image_array
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
    
    offset = 15
    crop_original_image = image_array[smallest_y:largest_y + offset, 
                                      smallest_x:largest_x + offset]
    crop_original_image = np.array(crop_original_image, np.float16)
    
    #resizes the image with a fixed width of 300 with respect to its aspect ratio 
    crop_image = Image.fromarray(np.uint8(crop_original_image))
    crop_width = 300
    percent = (crop_width / float(crop_image.size[0]))
    crop_height = int((float(crop_image.size[1]) * float(percent)))
    crop_image = crop_image.resize((crop_width, crop_height), Image.ANTIALIAS) #ANTIALIAS reserves quality
    crop_image = np.array(crop_image, np.float32)
    width, height = crop_image.shape[0], crop_image.shape[1]
    
    # white background to create the same shapes (needed for keras generator)
    pasted_crop = Image.new("RGB", (300, 300), color = "white")
    # creating an image from a PIL numpy array
    new_image = Image.fromarray(np.uint8(crop_image))
    pasted_crop.paste(new_image, (0, 0, height, width))
    # convert back to numpy array
    pasted_crop = np.array(pasted_crop, np.float32)
    
    return pasted_crop

def train_model(train_file = 'random_shapes_cropped.pkl', job_dir = './', 
                **args):
    # set the loggining path for ML Engine logging to storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))
    
    # need tensorflow to open file descriptor for google cloud to read
    with file_io.FileIO(train_file, mode='r') as f:
        # joblib loads compressed files consistenting of large datasets 
        save = joblib.load(f)
        train_shape_dataset = save['train_shape_dataset']
        train_y_dataset = save['train_y_dataset']
        validate_shape_dataset = save['validate_shape_dataset']
        validate_y_dataset = save['validate_y_dataset']
        del save  

    # using the functional API from keras
    input_data = Input(shape = (300, 300, 3))

    # Inception module architecture
    tower_1 = Conv2D(128, (6, 6), padding = 'same', activation = 'relu')(input_data)
    tower_1 = Conv2D(128, (6, 6), padding = 'same', activation = 'relu')(tower_1)
    tower_2 = Conv2D(128, (6, 6), padding = 'same', activation = 'relu')(input_data)
    tower_2 = Conv2D(128, (6, 6), padding = 'same', activation = 'relu')(tower_2)
    tower_3 = Conv2D(128, (6, 6), padding = 'same', activation = 'relu')(input_data)
    tower_3 = Conv2D(128, (6, 6), padding = 'same', activation = 'relu')(tower_3)

    output = concatenate([tower_1, tower_2, tower_3], axis = 1)
    flattened_output = Flatten()(output)

    fully_connected_layer = Dense(units = 512, activation = 'relu')(flattened_output)
    fully_connected_layer = Dropout(0.35)(fully_connected_layer) 
    fully_connected_layer = Dense(units = 512, activation = 'relu')(fully_connected_layer)
    fully_connected_layer = Dropout(0.35)(fully_connected_layer) 
    fully_connected_layer = Dense(units = 4, activation = 'softmax')(fully_connected_layer)

    model = Model(inputs = input_data, outputs = fully_connected_layer)
    # Compiling the CNN for single-label output: 
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

                       	
    datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2,
								 horizontal_flip = True) 
    validate_datagen = ImageDataGenerator(rescale = 1./255)
    validate_datagen.fit(validate_shape_dataset)
    validate_generator = validate_datagen.flow(validate_shape_dataset, validate_y_dataset, batch_size = 32)
        
    # early stopping prevent overfitting 
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2)
    datagen.fit(train_shape_dataset)
    train_generator = datagen.flow(train_shape_dataset, train_y_dataset, batch_size = 32)
    classifier.fit_generator(train_generator, 
                             steps_per_epoch = len(train_shape_dataset) / 32, 
                             epochs = 30,
                             callbacks = [early_stopping],
                             validation_data =  validate_generator,
                             validation_steps = 300)
    
    # evaluate the model 	               
    score = classifier.evaluate(validate_shape_dataset, validate_y_dataset, batch_size = 32, verbose = 0)
    print ("Test loss:", score[0])
    print ("Test accuracy", score[1])
    print ("Model Summary", classifier.summary())

    classifier.save('lstm_model.h5')

	# this makes predictions of the model
	# the model contains the model architecture and weights, specification of the chosen loss 
	# and optimization algorithm so that you can resume training if needed
    '''from keras.models import load_model
    model = load_model('lstm_model.h5')                       
    predictions = classifier.predict(validate_shape_dataset, batch_size = 32)    
    predictions[predictions >= 0.6] = 1
    predictions[predictions < 0.6] = 0     
    print ("Label predictions", predictions)      
    predict_score = f1_score(validate_y_dataset, predictions, average='macro')
    print("Prediction score", predict_score)'''     
              
    # Save the model to the Cloud Storage bucket's jobs directory
    with file_io.FileIO('model_lstm.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file',
                        help='local path of pickle file')
    parser.add_argument('--job-dir', 
                        help='Cloud storage bucket to export the model')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)


