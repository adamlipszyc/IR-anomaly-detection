import pandas as pd 
import numpy as np 
import random
import csv
import argparse
from evaluation.data_generation.utils import convert_end_to_traded
from utils import generate_real_positions, generate_random_zero_pairs

#THIS IS START, END 
PATTERN_1 = [[100, 0 , -100, -100], [100, 100, -100, 0], [100, 100, -100, -100]]
PATTERN_2 = [[100, 0, -100, -100], [100, 100, -100, -100], [100, 100, -100, 0]]
PATTERN_3 = [[100, 100, -100, 0], [100, 100, -100, -100], [100, 0, -100, -100]]
PATTERN_4 = [[100, 100, -100, 0], [100, 0, -100, -100], [100, 100, -100, -100]]
PATTERN_5 = [[100, 100, -100, -100], [100, 0, -100, -100], [100, 100, -100, 0]]
PATTERN_6 = [[100, 100, -100, -100], [100, 100, -100, 0], [100, 0, -100, -100]]

PATTERNS = [PATTERN_1, PATTERN_2, PATTERN_3, PATTERN_4, PATTERN_5, PATTERN_6]


NUM_EASY_SAMPLES = 100

NUM_REPEAT_SAMPLES = 100

REPEAT_FILENAME = "evaluation/data/repeat_scenarios.csv"

EASY_FILENAME = "evaluation/data/simple_scenarios.csv"

REPEAT_FILENAME_REAL = "evaluation/data/repeat_scenarios_with_real.csv"

EASY_FILENAME_REAL = "evaluation/data/simple_scenarios_with_real.csv"


def generate_examples_easy(with_real=False, patterns=PATTERNS, filename=""):
    file = EASY_FILENAME_REAL if with_real else EASY_FILENAME
    if filename:
        file = filename
    with open(file, "w") as file:
        writer = csv.writer(file)
        for i in range(NUM_EASY_SAMPLES):
            for pattern in patterns:
                scale = random.randint(1, 5)
                result = generate_random_zero_pairs(0, 10) 
                if with_real:
                    result += generate_real_positions(5,100) + generate_random_zero_pairs(3, 10)

                #CAN convert to numpy for speedup 
                for pair in pattern:
                    start1, end1, start2, end2= pair
                    result.append((start1 * scale, end1 * scale, start2 * scale, end2 * scale))
                    result += generate_random_zero_pairs(2, 50)
                
                if with_real:
                    result += generate_real_positions(5, 200)
                
                result = convert_end_to_traded(result)

                npresult = np.array([item for sublist in result for item in sublist]).flatten().astype(np.float64)
                if len(npresult) < 1100:
                    npresult = np.concatenate((npresult,np.array([0.0] * (1100 - len(npresult)))))
                writer.writerow(npresult)

def generate_examples_repeat(with_real=False, patterns=PATTERNS, filename=""):
    file = REPEAT_FILENAME_REAL if with_real else REPEAT_FILENAME
    if filename:
        file = filename

    metadata = [[]]
    with open(file, "w") as file:
        writer = csv.writer(file)
        for i in range(NUM_REPEAT_SAMPLES):
            
            initial_zeroes = generate_random_zero_pairs(0, 20)
            npresult = np.array(initial_zeroes).flatten()
            while len(npresult) != 1100:
                scale = random.randint(1, 5)
                index = np.round(np.random.uniform(0, len(PATTERNS) - 1)).astype(int)
                metadata[-1].append((index + 1, len(npresult)))
                result = []
                if with_real:
                    result = generate_random_zero_pairs(3, 10) + generate_real_positions(5,20) + generate_random_zero_pairs(3, 10)
                for pair in patterns[index]:
                    start1, end1, start2, end2 = pair
                    result.append((start1 * scale, end1 * scale, start2 * scale, end2 * scale))
                    result += generate_random_zero_pairs(2, 50)
                if with_real:
                    result += generate_real_positions(5,20) + generate_random_zero_pairs(3, 10)

                result = convert_end_to_traded(result)
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

    args = parser.parse_args()

    if not args.easy and not args.repeat:
        parser.error("You must specify at least one of --easy or --repeat or --scenario_4.")

    if args.easy:
        generate_examples_easy()
        generate_examples_easy(with_real=True)

    if args.repeat:
        generate_examples_repeat()
        generate_examples_repeat(with_real=True)
    

    






                
            














