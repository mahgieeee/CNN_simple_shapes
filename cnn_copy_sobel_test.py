"""
Created on Mon Nov 13 11:50:47 2017
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
@author: maggie
"""
from __future__ import print_function
from __future__ import division
import argparse
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime 
from tensorflow.python.lib.io import file_io 
import h5py
import joblib

"""to run code locally:
   python cnn_copy_sobel_test.py --job-dir ./ --train-file random_shapes_all.pkl 
"""
    
def train_model(train_file = 'random_shapes_png.pkl', job_dir = './', 
                #dropout_one = 0.2, dropout_two = 0.2, 
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
        del save  # help gc free up memory
        
    # Initializing the CNN by adding a simple sequential layer
    classifier = Sequential()

    # Step 1: 
    # Sequential layer consists of Convolution of type 3 by 3 convolutional 
    # window with 32 output filters for each input image uses relu layers 
    # to make layer less linear; the stride default is (1,1)
    # the default for padding is 'valid'
    classifier.add(Conv2D(128, (6, 6),
                          padding = 'valid', 
                          input_shape = (200, 200, 3), 
                          activation = 'relu'))

    # Step 2:  
    # Max Pooling downsamples the number pixels per neuron and create a max
    classifier.add(MaxPooling2D(pool_size = (6, 6)))

    # Adding a second convolutional layer, which is the same as the first one
    # the default for padding is 'valid'
    classifier.add(Conv2D(256, (6, 6), 
                          padding = 'valid', 
                          activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (6, 6)))


    # Step 3: Flattening the convolutional layers for input to MLP
    classifier.add(Flatten())
    
    # Step 4: 
    # Fully connected: Dense function is used to add a fully connected layer
    classifier.add(Dense(units = 256, activation = 'relu'))
    classifier.add(Dropout(0.35)) 
    #classifier.add(Dropout(dropout_one))
    
    # adding second hidden convolutional layer
    classifier.add(Dense(units = 256, activation = 'relu'))
    classifier.add(Dropout(0.25))
    #classifier.add(Dropout(dropout_two))
    
    classifier.add(Dense(units = 256, activation = 'relu'))
    classifier.add(Dropout(0.25))
    
    # softmax is an activation function for squashing probalities 
    # between 0-1, units represent number of output classes 
    classifier.add(Dense(units = 4, activation = 'softmax'))

    # Compiling the CNN for single-label output: 
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    classifier.compile(optimizer = adam, 
                       loss = 'categorical_crossentropy', 
                       metrics = ['accuracy'])
                       
    # Part 2: 
    # Feeding CNN the input images and fitting the CNN 
    # CNN uses data augmentation configuration to prevent overfitting
    datagen = ImageDataGenerator(rescale = 1./255, 
                                 shear_range = 0.2, 
                                 zoom_range = 0.2,
                                 horizontal_flip = True)
                         
    # augmentation configuration for rescaling images used for validation 
    validate_datagen = ImageDataGenerator(rescale = 1./255)
    
    validate_datagen.fit(validate_shape_dataset)
    validate_generator = validate_datagen.flow(validate_shape_dataset, 
                                               validate_y_dataset, 
                                               batch_size = 32)
       
    # early stopping prevent overfitting
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2)
    
    # compute quantities required for featurewise normalization
    datagen.fit(train_shape_dataset)
    
    # fits the model on batches with real-time data augmentation
    train_generator = datagen.flow(train_shape_dataset, 
                                   train_y_dataset, 
                                   batch_size = 32)
    classifier.fit_generator(train_generator, 
                             steps_per_epoch = len(train_shape_dataset) / 32, 
                             epochs = 50,
                             callbacks = [early_stopping], 
                             validation_data =  validate_generator, 
                             validation_steps = 300)     
                       
    # evaluate the model 	               
    score = classifier.evaluate(validate_shape_dataset, 
                                validate_y_dataset, 
                                batch_size = 32, 
                                verbose = 0)
    print ("Test loss:", score[0])
    print ("Test accuracy", score[1])
    print ("Model Summary", classifier.summary())

    classifier.save('model.h5')

    # Save the model to the Cloud Storage bucket's jobs directory
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file',
                        help='local path of pickle file')
    parser.add_argument('--job-dir', 
                        help='Cloud storage bucket to export the model')
    '''parser.add_argument('--dropout-one',
                        help='Dropout hyperparameter after the first dense layer')
    parser.add_argument('--dropout-two',
                        help='Dropout hyperparameter after the second dense layer')'''
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)


