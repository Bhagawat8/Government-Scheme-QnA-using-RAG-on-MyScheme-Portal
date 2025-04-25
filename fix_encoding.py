import pandas as pd
import chardet
def convert_encoding():
    # First detect encoding
    with open("cleaned_my_scheme_data.csv", 'rb') as f:
        rawdata = f.read(10000)  
        encoding = chardet.detect(rawdata)['encoding']
    
    print(f"Detected encoding: {encoding}")
    
    # Read with detected encoding
    df = pd.read_csv(
        "cleaned_my_scheme_data.csv",
        encoding=encoding,
        encoding_errors='replace'  # Correct parameter name
    )
    
    # Save with UTF-8
    df.to_csv("cleaned_my_scheme_data_fixed.csv", index=False, encoding='utf-8-sig')
    print("File saved with UTF-8 BOM encoding")

if __name__ == "__main__":
    convert_encoding()