import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from csv file")
    
    df['text'] = df.apply(lambda row: " \n ".join([f"{col}: {row[col]}" for col in df.columns]), axis=1)
    
    metadata_columns = ['tags', 'target_beneficiaries_states']
    
    for col in metadata_columns:
        if col not in df.columns:
            df[col] = ''
    
    df_loader = df[['text'] + metadata_columns]
    
    loader = DataFrameLoader(df_loader, page_content_column='text')
    documents = loader.load()
    
    for i, doc in enumerate(documents):
        doc.id = str(i)
    
    print(f"Converted {len(documents)} rows to documents")
    return documents