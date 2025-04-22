
import json
def prepare_portfolio_row(row):
    """
    Processes a row of portfolio data, extracting `start`, `traded` (start - end), and `ds_index`.
    
    :param row: Dictionary containing `name` and `portfolio_item` (a JSON object with `start`, `end`, and `ds_index`)
    :return: List of dictionaries with `start`, `traded`, and `ds_index`
    """
    name = row.get("name")
    portfolio = row.get("positions")
   
    
    if not isinstance(portfolio, list):
        raise ValueError("portfolio must be a list")
    
    positions = []
    for item in portfolio:
        
        start = item.get("start")
        end = item.get("end")
        ds_index = item.get("ds_index")
        
        if start is None or end is None or ds_index is None:
            raise ValueError("Missing required fields in portfolio")
        
        if not isinstance(start, str) or not isinstance(end, str):
            raise TypeError("Mismatch in types of the entries in database")


        traded = float(start) - float(end)
        positions.append({"start": float(start), "traded": traded})
        

        

    #Add appropriate padding
    if len(positions) != 550: #change to 570 if needed 
        leftover = 550 - len(positions) 
        for i in range(leftover):
            positions.append({"start": 0.0, "traded": 0.0})
    


    
    return {"name": name, "positions": positions}


if __name__ == "__main__":
    data = None

    with open("training_data/original_data/actual_positions.json", "r") as file:
        data = json.load(file)

        data = list(map(lambda x : prepare_portfolio_row(x), data))
        
    with open("training_data/original_data/prepped_data.json", "w") as file:
        json.dump(data, file, indent=4)