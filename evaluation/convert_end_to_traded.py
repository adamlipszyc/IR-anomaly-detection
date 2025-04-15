

def convert_end_to_traded(positions):
    """
    Converts a list of (start, end) tuples into a list of (start, traded) tuples
    """
    result = []
    for position in positions:
        for j in range(0, len(position), 2):
            start = position[j]
            end = position[j + 1]
            traded = float(start) - float(end)
            result.append((start, traded))

    return result 
