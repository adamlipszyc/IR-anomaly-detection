import random
import logging
import csv
import numpy as np
from typing import List, Tuple
from .config import SIMPLE_SCENARIOS, COMPLEX_SCENARIOS
from .scenarios.base import Scenario
from .utils import (
    generate_random_zero_pairs, 
    generate_real_positions, 
    convert_end_to_traded,
    call_until_success,
)
from ...log.utils import catch_and_log


class DataGenerator:
    """
    Generates simple and complex data entries for a collection of Scenario instances.
    """

    def __init__(
        self,
        scenarios: List[Scenario],
        num_simple: int = 1000,
        complex_iterations: int = 10,
        complex_samples: int = 100,
        vector_length: int = 550,
        real_limit: int = 200,
        logger: logging.Logger = None,
    ):
        self.scenarios = scenarios
        self.vector_length = vector_length
        self.real_limit = real_limit
        self.num_simple = num_simple
        self.complex_iterations = complex_iterations
        self.complex_samples = complex_samples
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @catch_and_log(Exception, "Saving entries to csv")
    def _save_entries_to_csv(self, entries, filename):
        with open(filename, "w") as file:
            writer = csv.writer(file)
            for entry in entries:
                writer.writerow(entry)
        
        self.logger.info("Entries saved to %s", filename)

    def _generate_simple_entry(self, scenario: Scenario, with_real: bool = False) -> List[Tuple[float, float]]:
        # Start with zeros
        entry = generate_random_zero_pairs(0, 30)
        # Optionally mix in some real positions early
        if with_real:
            real_count = random.randint(3, self.real_limit)
            entry += generate_real_positions(real_count)
            entry += generate_random_zero_pairs(0, 20)

        scale = random.randint(1, 5)
        # Insert the scenario
        data = call_until_success(scenario.generate)
        entry += [(start * scale, end * scale) for start, end in data]

        # Append a tail of zeros (and real, if flagged)
        tail = generate_random_zero_pairs(0, 30)
        if with_real:
            tail = generate_real_positions(random.randint(3, self.real_limit)) + tail
        entry += tail

        # Pad or trim to vector_length
        if len(entry) < self.vector_length:
            remaining = self.vector_length - len(entry)
            if with_real and random.choice([True, False]):
                entry += generate_real_positions(remaining)
            else:
                entry += [(0.0, 0.0)] * remaining
        else:
            entry = entry[: self.vector_length]

        return entry

    @catch_and_log(Exception, "Generating simple entries")
    def generate_simple(self) -> None:
        """
        Generate flattened, "simple" entries for each scenario.

        Returns:
            A list of flattened lists (length = vector_length * 2).
        """
        entries: List[List[float]] = []

        for i in range(self.num_simple):
            for scenario in self.scenarios:
                try:
                    with_real = random.choice([True, False])
                    raw = self._generate_simple_entry(scenario, with_real)
                    traded = convert_end_to_traded(raw)
                    # Flatten tuples to a single list of floats
                    flat = np.array(traded).flatten().astype(np.float64)
                    entries.append(flat)
                except Exception:
                    self.logger.exception(
                        "Failed to generate simple entry for scenario %s", scenario
                    )

        random.shuffle(entries)

        self._save_entries_to_csv(entries, SIMPLE_SCENARIOS)
    


    def _generate_complex_batch(
        self, 
        patterns: List[Scenario],
        with_real: bool = False
    ) -> List[List[float]]:
        entries: List[List[float]] = []
        for i in range(self.complex_samples):
            # initial zeros
            entry: List[Tuple[float, float]] = generate_random_zero_pairs(0, 20)
            # build until vector_length
            while len(entry) < self.vector_length:
                scale = random.randint(1, 5)
                choice = random.choice(patterns)
                block = []
                if with_real:
                    block += generate_random_zero_pairs(3, 10)
                    block += generate_real_positions(random.randint(5, 20))
                    block += generate_random_zero_pairs(3, 10)


                data = call_until_success(choice.generate)
                # scale the scenario block
                block += [(start * scale, end * scale) for start,end in data]

                if with_real:
                    block += generate_real_positions(random.randint(5, 20))
                    block += generate_random_zero_pairs(3, 10)

                # decide whether to pad or append
                if len(entry) + len(block) > self.vector_length:
                    remaining = self.vector_length - len(entry)
                    if with_real and random.choice([True, False]):
                        entry += generate_real_positions(remaining)
                    else:
                        entry += [(0.0, 0.0)] * remaining
                else:
                    entry += block

            traded = convert_end_to_traded(entry)
            flat = np.array(traded).flatten().astype(np.float64)
            entries.append(flat)

        return entries

    @catch_and_log(Exception, "Generating complex entries")
    def generate_complex(
        self, patterns_per_batch: int = 4
    ) -> None:
        """
        Generate "complex" entries by shuffling through scenario patterns.

        Args:
            num_iterations: how many times to shuffle-and-generate.
            patterns_per_batch: how many distinct scenarios per iteration.
        """
        all_entries: List[List[float]] = []
        for i in range(self.complex_iterations):
            with_real = random.choice([True, False])
            # pick a random subset of scenarios
            patterns = random.sample(self.scenarios, k=min(patterns_per_batch, len(self.scenarios)))
            batch = self._generate_complex_batch(patterns, with_real)
            all_entries.extend(batch)
        
        self._save_entries_to_csv(all_entries, COMPLEX_SCENARIOS)
        
