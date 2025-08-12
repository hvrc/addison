import random

def generate_validation_data(filename, num_samples=10000):
    with open(filename, 'w') as f:
        f.write('x1,x2,sum\n')
        for _ in range(num_samples):
            x1 = random.randint(0, 9)
            x2 = random.randint(0, 9)
            sum_val = x1 + x2
            f.write(f'{x1},{x2},{sum_val}\n')

if __name__ == '__main__':
    generate_validation_data('../data/validation_data.csv')