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

# Load data
ancient_df = pd.read_csv('ancient_mtDNA_metadata.csv')
modern_df = pd.read_csv('modern_mtDNA_metadata.csv')

def balance_by_column(df_a, df_b, column_name):
    df_a['set'] = 'a'
    df_b['set'] = 'b'
    combined_df = pd.concat([df_a, df_b], ignore_index=True)

    # Balance a and b for each group 
    balanced_list_a = []
    balanced_list_b = []

    # Get the list of unique continents
    groups = combined_df[column_name].unique()

    for group in groups:
        # Get all samples for this specific continent
        group_data = combined_df[combined_df[column_name] == group]
        
        # Split into Ancient and Modern
        samples_a = group_data[group_data['set'] == 'a']
        samples_b = group_data[group_data['set'] == 'b']
        
        # Find the smaller count (the bottleneck)
        count_a = len(samples_a)
        count_b = len(samples_b)
        k = min(count_a, count_b)
        
        if k > 0:
            # Sample k from each and add to our list
            balanced_a = samples_a.sample(n=k, random_state=42)
            balanced_b = samples_b.sample(n=k, random_state=42)
            balanced_list_a.append(balanced_a)
            balanced_list_b.append(balanced_b)
            print(f"Column '{group}': Found {count_a} in a, {count_b} in b. Keeping {k} of each.")
        else:
            print(f"Column '{group}': Dropped (Missing in either a or b).")

    if balanced_list_a and balanced_list_b:
        balanced_df_a = pd.concat(balanced_list_a, ignore_index=True)
        balanced_df_b = pd.concat(balanced_list_b, ignore_index=True)

        balanced_df_a.drop(columns=['set'])
        balanced_df_b.drop(columns=['set'])

        df_a.drop(columns=['set'])
        df_b.drop(columns=['set'])

        return balanced_df_a, balanced_df_b
    
ancient_df_balanced, modern_df_balanced = balance_by_column(ancient_df, modern_df, 'continent')

# Final verification
print("\n--- Final Balanced Counts ---")

# Get counts for each
anc_counts = ancient_df_balanced.groupby('continent').size()
mod_counts = modern_df_balanced.groupby('continent').size()

# Combine them into a single DataFrame for a side-by-side view
stats = pd.concat([anc_counts, mod_counts], axis=1).fillna(0)
stats.columns = ['Ancient', 'Modern']

print(stats)

# Execute splits
anc_train, anc_val, anc_test = split_three_way(ancient_df_balanced, "Ancient")
mod_train, mod_val, mod_test = split_three_way(modern_df_balanced, "Modern")
