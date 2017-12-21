from __future__ import print_function
from __future__ import division
from keras.models import load_model
import argparse
#from sklearn.metrics import f1_score
from datetime import datetime 
from tensorflow.python.lib.io import file_io 
import h5py
import joblib

"""to run code locally:
   python test_model.py --job-dir ./ --train-file test_random_shapes.pkl 
"""
    

def train_model(train_file = 'test_random_shapes.pkl',
                job_dir = './', 
                **args):
    # set the loggining path for ML Engine logging to storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))
    
    # need tensorflow to open file descriptor for google cloud to read
    with file_io.FileIO(train_file, mode='r') as f:
        # joblib loads compressed files consistenting of large datasets 
        save = joblib.load(f)
        test_shape_dataset = save['train_shape_dataset']
        test_y_dataset = save['train_y_dataset']
        del save  # help gc free up memory

    # this makes predictions of the model
    # the model contains the model architecture and weights, specification of the chosen loss 
    # and optimization algorithm so that you can resume training if needed
    model = load_model('model.h5')                       
    '''predictions = model.predict(test_shape_dataset, batch_size = 32)    
    predictions[predictions >= 0.6] = 1
    predictions[predictions < 0.6] = 0     
    print ("Label predictions", predictions)      
    predict_score = f1_score(test_y_dataset, predictions, average='macro')
    print("Prediction score", predict_score)'''
    # evaluate the model 	               
    score = model.evaluate(test_shape_dataset, 
                           test_y_dataset, 
                           batch_size = 32, 
                           verbose = 1)
    print ("Test loss:", score[0])
    print ("Test accuracy", score[1])
    print ("Model Summary", model.summary())
     
             


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


