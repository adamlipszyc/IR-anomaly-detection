import random 
import numpy as np
import csv
import generate_scenario_4
import generate_scenario_9
import generate_scenario_10
import generate_scenario_11
from generate_scenarios import generate_zero_pairs, generate_real_positions
import argparse
from convert_end_to_traded import convert_end_to_traded

CHOICES = [(4,generate_scenario_4.generate_test_case), (9,generate_scenario_9.generate_scenario), 
           (10,generate_scenario_10.generate_scenario), (11,generate_scenario_11.generate_scenario)]

NUM_SIMPLER_SAMPLES = 100

NUM_COMPLEX_SAMPLES = 10

NUM_ITERATIONS_COMPLEX = 100


def save_entries_to_csv(entries, filename):
    with open(filename, "w") as file:
        writer = csv.writer(file)
        for entry in entries:
            result = np.array(entry).flatten()
            writer.writerow(result)



def generate_positions(with_real=False, limit_real=200):
    result = generate_zero_pairs(0, 30)
    if with_real:
        result += generate_real_positions(random.randint(3, limit_real)) + generate_zero_pairs(0, 20)

    return result

def generate_simple_entry(scenario, with_real=False):

    result = generate_positions(with_real, limit_real=80)

    result += scenario 

    result += generate_positions(with_real)

    if len(result) < 550:
        zeroed = random.randint(0, 1)
        num_left = (550 - len(result))
        if with_real and not zeroed:
            result += generate_real_positions(num_left)
            return result 
        
        result += [(0.0, 0.0)] * (550 - len(result))
    
    return result


def convert_scenarios_to_simple_entry():
    for scen_index, choice in CHOICES:
        entries = []
        for i in range(NUM_SIMPLER_SAMPLES):
            correct = False
            while not correct:
                try:
                    scenario = choice()
                    correct = True
                except:
                    pass

            zero_entry = generate_simple_entry(scenario)
            real_entry = generate_simple_entry(scenario, with_real=True)
            entries.append(convert_end_to_traded(zero_entry))
            entries.append(convert_end_to_traded(real_entry))
        
        save_entries_to_csv(entries, f"evaluation/data/simple_scenario_{scen_index}.csv")

            


def convert_complex_scenarios(scenarios, with_real=False):
    metadata = [[]]
    entries = []
    print(scenarios[0])
    for i in range(NUM_COMPLEX_SAMPLES):
        initial_zeroes = generate_zero_pairs(0, 20)
        entry = initial_zeroes
        while len(entry) != 550:
            scale = random.randint(1, 5)
            index = random.randint(0, len(scenarios) - 1)
            metadata[-1].append((index + 1, len(entry)))
            result = []
            if with_real:
                result = generate_zero_pairs(3, 10) + generate_real_positions(random.randint(5,20)) + generate_zero_pairs(3, 10)

            result += list(map(lambda pair: (pair[0] * scale, pair[1] * scale), scenarios[index]))
            
            if with_real:
                result += generate_real_positions(random.randint(5,20)) + generate_zero_pairs(3, 10)
            
            if len(entry) + len(result) > 550:
                zeroed = random.randint(0, 1)
                num_left = (550 - len(entry))
                if with_real and not zeroed:
                    entry += generate_real_positions(num_left)
                else:
                    entry += [(0.0, 0.0)] * (550 - len(entry))
            else:
                entry += result


        entries.append(convert_end_to_traded(entry))
        metadata.append([])
    
   

    print(metadata)
    return entries          



def convert_scenarios_to_complex_entry():
    options = [i for i, _ in enumerate(CHOICES)]
    entries = []
    for i in range(NUM_ITERATIONS_COMPLEX):
        random.shuffle(options)
        patterns = []
        for j in options:
            correct = False
            while not correct:
                try:
                    scenario = CHOICES[j][1]()
                    correct = True
                except:
                    pass
            patterns.append(scenario)
        entries += convert_complex_scenarios(patterns)
        entries += convert_complex_scenarios(patterns, with_real=True)

        print(i)

    save_entries_to_csv(entries, "evaluation/data/complex_entries.csv")
    




    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--simple', action='store_true')
    parser.add_argument('-c',  '--complex', action='store_true')

    args = parser.parse_args()

    if not args.simple and not args.complex:
        parser.error("You must specify at least one of --simple, ... --complex.")

    if args.simple:
        convert_scenarios_to_simple_entry()

    if args.complex:
        convert_scenarios_to_complex_entry()
