# Code demonstrates how to perform a minibatch technique in vanilla python
# Minibatching is extremely useful when processing large tensors. 
# Many machine learning frameworks provide functions that make this process opaque. 
# Below is a simple example that uses basic python libraries to split large datasets into manageable batches

# Import pprint for more friendly console output
from pprint import pprint

# Define a test feature set
# In this example we want to illustrate the mini batching technique
# hence we choose 9 and call the function later with mini batches of three
features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F44'],
    ['F41','F42','F43','F44'],
    ['F51','F52','F53','F54'],
    ['F61','F62','F63','F64'],
    ['F71','F72','F73','F74'],
    ['F81','F82','F83','F84'],
    ['F91','F92','F93','F94']
]

# Define a test label set
# The key thing to remember is that the features and labels are the same length
# It is easy to imagine that these labels are for a simple classification problem
# like figuring out something is or isn't in a particular classification
labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42'],
    ['L51','L52'],
    ['L61','L62'],
    ['L71','L72'],
    ['L81','L82'],
    ['L91','L92']
]

# Define a function that breaks up a large feature/label set into a more manageable
# The function will take a batch size parameter, the features, and labels of
# a data set.
def batches(batch_size, features, labels):
    # Add an assertion that our two matrix values will be the same length
    assert len(features) == len(labels)

    # Create an empty array for the output
    output_batches = []
    # Define the sample size. It is equal to the feature length
    sample_size = len(features)

    # Loop through the full feature/label set and process them according
    # to the batch size. In this example the batch size will be 3 which should
    # produce 3 mini batches from the feature set.
    # Use the range() sequence with the start, stop, step parameters
    # For more information visit the Python docs at https://docs.python.org/3/library/stdtypes.html#range
    for start_i in range(0, sample_size, batch_size):

        # end_i will be our end of our batch range. Increment the start_i
        # by the batch size until we
        end_i = start_i + batch_size

        # Each batch consists of the records between the start_i and end_i indexes
        batch = [features[start_i:end_i], labels[start_i:end_i]]

        # Finally, append the newly created batch to the output batch array
        output_batches.append(batch)

    # Once the operation is complete, return the output batch
    return output_batches

# To see what the feature/label data set looks like
# call pprint to view the values
pprint(batches(3,features,labels))
