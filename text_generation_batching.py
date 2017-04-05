# Import the numpy library to handle n-dimensional arrays
import numpy as np

'''
    Recommended reading: https://karpathy.github.io/2015/05/21/rnn-effectiveness/
    Text generation has a unique requirement for creating batches.
    We want to predict what word will come next after training on a given corpus.
    This requires a breakdown of the input data, (list of words that have been vectorized)
    into batches of steps. 
'''

# Declare variables for testing purposes. 
# Notice the list comprehension used to generate a list of numbers. 
int_text = [x for x in range(70000)]
batch_size = 2
seq_length = 3

# Define a function that takes care of stepping through the inputs
def step(starting_position, seq_length, inputs):

    idx = starting_position
    window = []
    
    for n in range(((len(inputs)) - 1) // seq_length):

        window.append(inputs[idx:(idx + seq_length)])
        idx += seq_length  
        
    return window

# Define a function that takes creates batches for inputs and targets that are appropriately stepped

def get_batches(int_text, batch_size, seq_length):
    
    inputs = step(0, seq_length, int_text)
    targets = step(1, seq_length, int_text)
 
    batch = np.array([inputs, targets], np.int32)
    
    print('Batch: ---- \nInputs:\n{}\nTargets:\n{}'.format(batch[0][0::10000], batch[1][0::10000]))
    
# Call the function   
get_batches(int_text, batch_size, seq_length)
