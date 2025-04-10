import random
import csv

def generate_trade_block(num_trades=5):
    """Generate a block of positive trade positions with End randomly less than or equal to Start."""
    trade_value_choices = [100 * i for i in range(1, 8)]  # 100 to 400
    block = []
    total = 0
    large_negative_index = random.randint(0, num_trades - 1)
    for i in range(num_trades):
        start = random.choice(trade_value_choices)
        shouldTrade = random.randint(0, 1)
        if shouldTrade:
            end = random.randint(0, start)
        else:
            end = start
        block.append((start, end))
        total += (start - end)
        if large_negative_index == i:
            large_value = round(random.randint(1000, 3000) / 100) * 100
            block.append((-large_value, -large_value))
    return block, total

def balance_traded_amounts(positive_block, negative_block, positive_sum, negative_sum):
    """Ensure the sum of positive traded equals negative traded by reducing the excess in reverse order."""
    
    larger_sum = 0
    smaller_sum = 0
    larger_block = None

    # Determine which is the larger block regarding amount traded
    if positive_sum > negative_sum:
        larger_sum = positive_sum
        smaller_sum = negative_sum
        larger_block = positive_block
    else:
        larger_sum = negative_sum
        smaller_sum = positive_sum
        larger_block = negative_block
    

    # Balance the amounts traded so they match 
    excess = larger_sum - smaller_sum
    i = len(larger_block) - 1
    while excess != 0:
        #Skip the large negative position
        if abs(larger_block[i][0]) >= 1000:
            i -= 1 
            continue

        # Figures out much we can possibly reduce by 
        reducible = larger_block[i][0] - larger_block[i][1]
        reduce_by = min(excess, abs(reducible))
        if reducible < 0 and reduce_by > 0:
            reduce_by = -reduce_by
        
        # Carry out reduction 
        larger_block[i] = (larger_block[i][0], larger_block[i][1] + reduce_by)
        excess -= abs(reduce_by)
        i -= 1

    return positive_block, negative_block

def generate_zero_rows(count):
    return [[0, 0] for _ in range(count)]

def generate_scenario():
    """Generate one full test scenario with two mismatched blocks and intermediate zero rows."""
    pos_block, positive_trade_sum = generate_trade_block(random.randint(3, 6))
    neg_block, negative_trade_sum = generate_trade_block(random.randint(3, 6))

    neg_block = [(-start, -end) for start, end in neg_block]

    pos_block, neg_block = balance_traded_amounts(pos_block, neg_block, positive_trade_sum, negative_trade_sum)

    # Insert random number of zeros in between blocks
    zero_rows = generate_zero_rows(random.randint(2, 6))



    return generate_zero_rows(random.randint(1, 6)) + pos_block + zero_rows + neg_block + generate_zero_rows(random.randint(1, 6))

def generate_multiple_scenarios(num=10):
    all_rows = []
    for _ in range(num):
        scenario = generate_scenario()
        all_rows.extend(scenario)
        all_rows.append([-999999999, -999999999])  # delimiter
    return all_rows

def write_to_csv(rows, filename="evaluation/data/generated_scen_9.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Start", "End"])
        writer.writerows(rows)

# Example usage
if __name__ == "__main__":
    data = generate_multiple_scenarios(10)
    write_to_csv(data)
