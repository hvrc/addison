import random

# we generate 10,000 rows of data,
# each has two random single digit numbers as the inputs x1 and x2,
# and their sum as the output y
# obviously, since there are only 81 unique combinations of all single digit numbers,
# some combinations will be repeated
# how does the concept of repeated data affect the training of a neural network?

def generate_data(filename, num_samples=10000):
    with open(filename, 'w') as f:
        f.write('x1,x2,y\n')
        for _ in range(num_samples):
            x1 = random.randint(0, 9)
            x2 = random.randint(0, 9)
            y = x1 + x2
            f.write(f'{x1},{x2},{y}\n')

generate_data('../data/data.csv')