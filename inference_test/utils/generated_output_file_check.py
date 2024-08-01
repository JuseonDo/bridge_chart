import os

def generated_output_check(output_save_path:str):
    if os.path.exists(output_save_path):
        with open(output_save_path) as f:
            predictions = [line.strip().replace("[[SEP]]","\n") for line in f]
        start_idx = len(predictions)
        print(f"Already saved outputs {start_idx} line exits.")
        
    else:
        print(f"Output file not found.")
        predictions = []
        start_idx = len(predictions)
    
    return predictions, start_idx