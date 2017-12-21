from __future__ import print_function
import pickle
import joblib
from multiprocessing import Pool

def load_train_or_test(files):    
    with open(files, 'rb') as f:
        try:
            print ("Opening files")
            print (files)
            while True:
                yield pickle.load(f, encoding = 'bytes') # python version 2.7
        except EOFError:
            pass
    
def get_data(item):
    return item
    
circle_dataset = []   
def result(data): 
    circle_dataset.append(data)
    
triangle_dataset = []   
def result1(data): 
    triangle_dataset.append(data)

rectangle_dataset = []   
def result2(data): 
    rectangle_dataset.append(data)
    
square_dataset = []   
def result3(data): 
    square_dataset.append(data)
 
valid_circle_dataset = []   
def result4(data): 
    valid_circle_dataset.append(data)

valid_triangle_dataset = []   
def result5(data): 
    valid_triangle_dataset.append(data)

valid_rectangle_dataset = []   
def result6(data): 
    valid_rectangle_dataset.append(data)

valid_square_dataset = []   
def result7(data): 
    valid_square_dataset.append(data)
    
def load(result, pickle_file):
    shape_loader = Loader()
    shape_loader.run (result, pickle_file)
    
class Loader(object):
    
    def run(self, load_result, filename):
        p = Pool(processes=4)
        for item in load_train_or_test(filename):
            p.apply_async(get_data, (item,), callback = eval(load_result))
        p.close()
        p.join()    
        
        
if __name__ == '__main__':
    
    load('result', 'circle.pkl')
    load('result1', 'triangle.pkl')
    load('result2', 'rectangle.pkl')
    load('result3', 'square.pkl')
    '''load('result4', 'valid_circle.pkl')
    load('result5', 'valid_triangle.pkl')
    load('result6', 'valid_rectangle.pkl')
    load('result7', 'valid_square.pkl')'''
    
    #merge dataset
    '''train_data = (circle_dataset + triangle_dataset + rectangle_dataset + 
                 square_dataset)
    validation_data = (valid_circle_dataset + valid_triangle_dataset + 
                       valid_rectangle_dataset + valid_square_dataset)'''
    
    pickle_file = 'data_shapes.pkl'
    try:
        f = open(str(pickle_file), 'wb')
        save = {#'train_data': train_data,
                #'validation_data': validation_data,
                'circle_dataset': circle_dataset,
                'triangle_dataset': triangle_dataset,
                'rectangle_dataset': rectangle_dataset,
                'square_dataset': square_dataset,
                #'valid_circle_dataset': valid_circle_dataset,
                #'valid_triangle_dataset': valid_triangle_dataset,
                #'valid_rectangle_dataset': valid_rectangle_dataset,
                #'valid_square_dataset': valid_square_dataset
               }
        joblib.dump(save, f, compress = True)
        f.close()
    except Exception as e:
        print('Unable to save data to', str(pickle_file), ':', e)
        raise
    
    