import os

def get_lines(filename):
    """
    Reads filename (a text file) and returns the lines of text as a list.
    """
    with open(filename, "r") as f:
        return f.readlines()

def preprocess_text_with_line_numbers(filename):
    """
    Returns a list of dictionaries of abstract line data.
    """
    input_lines = get_lines(filename)
    abstract_lines = "" 
    abstract_samples = [] 
    
    for line in input_lines:
        if line.startswith("###"):
            abstract_id = line
            abstract_lines = "" 
        elif line.isspace(): 
            abstract_line_split = abstract_lines.splitlines() 
            
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abstract_line.split("\t") 
                line_data["target"] = target_text_split[0] 
                line_data["text"] = target_text_split[1].lower() 
                line_data["line_number"] = abstract_line_number 
                line_data["total_lines"] = len(abstract_line_split) - 1 
                abstract_samples.append(line_data)
        else: 
            abstract_lines += line
            
    return abstract_samples

def load_data(data_dir="pubmed_rct"):
    """
    Loads train, validation, and test data from the data directory.
    """
    train_path = os.path.join(data_dir, "train.txt")
    val_path = os.path.join(data_dir, "dev.txt")
    test_path = os.path.join(data_dir, "test.txt")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Data files not found in " + data_dir)

    print("Loading and preprocessing data...")
    train_samples = preprocess_text_with_line_numbers(train_path)
    val_samples = preprocess_text_with_line_numbers(val_path)
    test_samples = preprocess_text_with_line_numbers(test_path)
    
    return train_samples, val_samples, test_samples
