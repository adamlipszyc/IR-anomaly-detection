import random 
import numpy as np
import csv
import generate_scenario_4
import generate_scenario_9
import generate_scenario_10
import generate_scenario_11
from generate_scenarios import generate_examples_easy, generate_examples_repeat


CHOICES = [generate_scenario_4.generate_test_case, generate_scenario_9.generate_scenario, 
           generate_scenario_10.generate_scenario, generate_scenario_11.generate_scenario]

NUM_SAMPLES = 100

NUM_REPEAT_SAMPLES = 100

def convert_complex_scenarios(scenarios, filename):
    with open(filename, "w") as file:
        writer = csv.writer(file)
        for scenario in scenarios:
            result = np.array(scenario).flatten()
            writer.writerow(result)
            

    for generator in CHOICES:
        generate_examples_easy

def convert_scenarios_to_entry():


    options = [i for i, _ in enumerate(CHOICES)]
    for i in range(NUM_REPEAT_SAMPLES):
        random.shuffle(options)
        patterns = [CHOICES[i]() for i in options]
        generate_examples_repeat(with_real=True, patterns=patterns, filename="evaluation/data/complex_scen_mixed.csv")



    random.randint(0,len(CHOICES) - 1)

    

