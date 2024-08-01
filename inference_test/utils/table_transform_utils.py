import pandas as pd
from io import StringIO

def convert_value(value:str):
    if not isinstance(value,str): return value
    if '.' in value or 'e' in value or 'E' in value:
        try: return float(value)
        except: return value
    else:
        try: return int(value)
        except: return value

def transform_to_dict(table):
    """
    table should be csv format.
    """
    df = pd.read_csv(StringIO(table), sep=",")
    index_column = df.columns[0]
    result = {}
    for col in df.columns[1:]:
        result[col] = df.set_index(index_column)[col].apply(convert_value).to_dict()
    result_strings = [f"{col} = {result[col]}" for col in result]
    
    return '\n'.join(result_strings)


def transform_to_csv(table:str):
    table = '_,' + ','.join(table.split('\n')[0].split(',')[1:]).strip() + '\n' + '\n'.join(table.split('\n')[1:]).strip()
    return table


# table = "id,A,B,C,D\n1,3.14,Hello,7,2023\n2,2.71,World,8,2024\n3,1.61,Test,9,2025\n4,0.57,Sample,10,2026\n"
# formatted_result = transform_to_csv(table)
# print(formatted_result)

# formatted_result = transform_to_dict(table)
# print(formatted_result)


