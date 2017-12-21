"""
Program tests cnn performance with different cropped input shapes by using 
the Spatial Pyramid Pooling Layer https://github.com/yhenon/keras-spp/blob/master/spp
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
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from spp.SpatialPyramidPooling import SpatialPyramidPooling
#from sklearn.metrics import f1_score
import numpy as np
from datetime import datetime 
from PIL import Image
from tensorflow.python.lib.io import file_io 
import h5py
import joblib
import numpy as np

"""to run code locally:
   python cnn_sobel_spp.py --job-dir ./ --train-file random_shapes_cropped.pkl
"""
    
def train_model(train_file = 'random_shapes_cropped.pkl', job_dir = './', **args):
    # set the loggining path for ML Engine logging to storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))
    
    # need tensorflow to open file descriptor for google cloud to read
    with file_io.FileIO(train_file, mode='r') as f:
        # joblib loads compressed files consistenting of large datasets 
        save = joblib.load(f)
        train_shape_dataset = save['train_shape_dataset']#300 by 300 cropped
        train_y_dataset = save['train_y_dataset']
        validate_shape_dataset = save['validate_shape_dataset']
        validate_y_dataset = save['validate_y_dataset']
        train_shape_dataset1 = save['train_shape_dataset1'] #300 by 150 cropped
        train_y_dataset1 = save['train_y_dataset1']
        validate_shape_dataset1 = save['validate_shape_dataset1']
        validate_y_dataset1 = save['validate_y_dataset1']
        del save 
         
    classifier = Sequential()
    
    classifier.add(Conv2D(64, (6, 6), 
                          padding = 'valid', 
                          input_shape = (None, None, 3), 
                          activation = 'relu'))
    classifier.add(Conv2D(128, (6, 6), padding = 'valid', activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (6, 6)))
    classifier.add(Dropout(0.1)) 
    classifier.add(SpatialPyramidPooling([1, 2, 4]))
    classifier.add(Dense(units = 512, activation = 'relu'))
    classifier.add(Dropout(0.35)) 

    # softmax is an activation function for squashing probalities 
    # between 0-1, units represent number of output classes 
    classifier.add(Dense(units = 4, activation = 'softmax'))

    # Compiling the CNN for single-label output: 
    classifier.compile(optimizer = 'adam', 
                       loss = 'categorical_crossentropy', 
                       metrics = ['accuracy'])
    # fitting model to 300 by 300 pixels shape
    classifier.fit(train_shape_dataset, 
                   train_y_dataset, 
                   epochs = 30, 
                   validation_data = (validate_shape_dataset, validate_y_dataset),
                   steps_per_epoch = 32,
                   validation_steps = 100)
    
    # fitting model to 300 by 150 pixels shape
    classifier.fit(train_shape_dataset1, 
                   train_y_dataset1, 
                   epochs = 30, 
                   validation_data = (validate_shape_dataset1, validate_y_dataset1),
                   steps_per_epoch = 32,
                   validation_steps = 100)

    classifier.save('spp_model.h5')

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


