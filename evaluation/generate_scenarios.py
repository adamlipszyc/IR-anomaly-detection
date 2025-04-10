import pandas as pd 
import numpy as np 
import random
import csv
import argparse

PATTERN_1 = [[100, 0 , -100, -100], [100, 100, -100, 0], [100, 100, -100, -100]]
PATTERN_2 = [[100, 0, -100, -100], [100, 100, -100, -100], [100, 100, -100, 0]]
PATTERN_3 = [[100, 100, -100, 0], [100, 100, -100, -100], [100, 0, -100, -100]]
PATTERN_4 = [[100, 100, -100, 0], [100, 0, -100, -100], [100, 100, -100, -100]]
PATTERN_5 = [[100, 100, -100, -100], [100, 0, -100, -100], [100, 100, -100, 0]]
PATTERN_6 = [[100, 100, -100, -100], [100, 100, -100, 0], [100, 0, -100, -100]]

PATTERNS = [PATTERN_1, PATTERN_2, PATTERN_3, PATTERN_4, PATTERN_5, PATTERN_6]

ZERO_PAIR = [0, 0]

NUM_EASY_SAMPLES = 100

NUM_REPEAT_SAMPLES = 100

def generate_zero_pairs(lower, upper):
    num_zeroes = np.round(np.random.uniform(lower, upper)).astype(int)
    return [ZERO_PAIR for _ in range(num_zeroes)]

def generate_real_positions(num_positions=20, value_range=(-500, 500)):
    return [
        (v := random.randint(*value_range) // 10 * 10, v)  # round to nearest 10
        for _ in range(num_positions)
    ]

def generate_examples_easy(with_real=False):
    with open("evaluation/data/simple_scenarios.csv", "w") as file:
        writer = csv.writer(file)
        for i in range(NUM_EASY_SAMPLES):
            for pattern in PATTERNS:
                result = []
                if with_real:
                    result = generate_zero_pairs(3, 10) + generate_real_positions(random.randint(5,20)) + generate_zero_pairs(3, 10)
                for pair in pattern:
                    result.append(pair)
                    result += generate_zero_pairs(2, 50)
                
                if with_real:
                    result += generate_real_positions(random.randint(5, 20))

                npresult = np.array([item for sublist in result for item in sublist]).flatten().astype(np.float64)
                if len(npresult) < 1100:
                    npresult = np.concatenate((npresult,np.array([0.0] * (1100 - len(npresult)))))
                writer.writerow(npresult)

def generate_examples_repeat(with_real=False):
    metadata = [[]]
    with open("evaluation/data/repeat_scenarios.csv", "w") as file:
        writer = csv.writer(file)
        for i in range(NUM_REPEAT_SAMPLES):
            
            initial_zeroes = generate_zero_pairs(0, 20)
            npresult = np.array(initial_zeroes).flatten()
            while len(npresult) != 1100:
                index = np.round(np.random.uniform(0, 5)).astype(int)
                metadata[-1].append((index + 1, len(npresult)))
                result = []
                if with_real:
                    result = generate_zero_pairs(3, 10) + generate_real_positions(random.randint(5,20)) + generate_zero_pairs(3, 10)
                for pair in PATTERNS[index]:
                    result.append(pair)
                    result += generate_zero_pairs(2, 50)
                if with_real:
                    result += generate_real_positions(random.randint(5,20)) + generate_zero_pairs(3, 10)
                flattened_results =  np.array([item for sublist in result for item in sublist]).flatten()
               
                if len(npresult) + len(flattened_results) > 1100:
                    npresult = np.concatenate((npresult, np.array([0.0] * (1100 - len(npresult)))))
                else:
                    npresult = np.concatenate((npresult, flattened_results))

            writer.writerow(npresult)

            metadata.append([])
    
    print(metadata)






if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--easy', action='store_true')
    parser.add_argument('-r', '--repeat', action='store_true')
    parser.add_argument('-4', '--scenario_4', action='store_true')

    args = parser.parse_args()

    if not args.easy and not args.repeat and not args.scenario_4:
        parser.error("You must specify at least one of --easy or --repeat or --scenario_4.")

    if args.easy:
        generate_examples_easy(with_real=True)

    if args.repeat:
        generate_examples_repeat(with_real=True)
    
    if args.scenario_4:
        pass

    






                
            














