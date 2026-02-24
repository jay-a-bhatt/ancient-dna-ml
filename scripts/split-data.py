import pandas as pd
from sklearn.model_selection import train_test_split

def split_three_way(df, label):
    # 1. First split: 60% Train, 40% 'Remainder'
    train, remainder = train_test_split(
        df, 
        test_size=0.4, 
        random_state=42, 
        stratify=df['continent']
    )
    
    # 2. Second split: Split the 40% Remainder into two equal 20% halves
    # Since 0.5 of 40% is 20%, we use test_size=0.5
    val, test = train_test_split(
        remainder, 
        test_size=0.5, 
        random_state=42, 
        stratify=remainder['continent']
    )
    
    print(f"--- {label} Split Summary ---")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

# Load your data
ancient_df = pd.read_csv('ancient_mtDNA_metadata.csv')
modern_df = pd.read_csv('modern_mtDNA_metadata.csv')

# Execute splits
anc_train, anc_val, anc_test = split_three_way(ancient_df, "Ancient")
mod_train, mod_val, mod_test = split_three_way(modern_df, "Modern")
