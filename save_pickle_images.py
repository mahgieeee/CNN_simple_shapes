"""
Created on Sun Nov 12 18:41:03 2017
Program loads pickle files and saves them as images in a folder 
@author: maggie
"""
from __future__ import print_function
import scipy.misc
from tensorflow.python.lib.io import file_io 
import joblib
import argparse
'''
to execute:
python save_pickle_images.py --train-file save_shapes.pkl --path --num-circles --num-triangles --num-rectangles --num-squares
'''
def save_dataset(dataset, num, path):
    name = []
    for i in range(num):
        name.append(path + str(i) + ".png")    

    for x in range(num):
        print(name[x])
        scipy.misc.imsave(name[x], dataset[x])

# num and path are command line argument 
# num is the number of images that will be saved with path as the file location        
def joblib_load(train_file, path, num_circles, num_triangles, num_rectangles, num_squares):
    with file_io.FileIO(train_file , mode='rb') as f: #open in rb for python 3
        save = joblib.load(f)
        circle_dataset = save['circle_dataset'] 
        triangle_dataset = save['triangle_dataset']
        rectangle_dataset = save['rectangle_dataset']
        square_dataset = save['square_dataset']
        #validation_all = save['validation_all']
        del save  # hint to help gc free up memory  
        
        save_dataset(circle_dataset, num_circles, path + 'circle_')
        save_dataset(triangle_dataset, num_triangles, path + 'triangle_')
        save_dataset(rectangle_dataset, num_rectangles, path + 'rectangle_')
        save_dataset(square_dataset, num_squares, path + 'square_')
        
if __name__ == '__main__':
      # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-squares', 
                        help='num of files for squares')
    parser.add_argument('--num-rectangles', 
                        help='num of files for rectangle')
    parser.add_argument('--num-triangles', 
                        help='num of files for triangle')
    parser.add_argument('--num-circles', 
                        help='num of files for circle')
    parser.add_argument('--path', 
                        help='pathname to be saved')
    parser.add_argument('--train-file',
                        help='local path of pickle file')
    args = parser.parse_args()
    arguments = args.__dict__
    joblib_load(**arguments)
    
    #example    
    #save_dataset(circle_dataset, 516, 'resized_training_set/valid_circle/circle_')
    #save_dataset(triangle_dataset, 285, 'resized_training_set/valid_triangle/triangle_')    
    #save_dataset(rectangle_dataset, 299, 'resized_training_set/valid_rectangle/rectangle_')
    #save_dataset(square_dataset, 224, 'resized_training_set/valid_square/square_')
 

