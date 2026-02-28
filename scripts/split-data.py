from types import MethodDescriptorType
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

fasta_data_filepaths = [
        "../data/modern_mtdna_raw.fasta",
        "../data/aadr.fasta",
        "../data/amtdb_2025.fasta"
]

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

def create_mega_fasta(ancient_df, modern_df, output_name):
    combined_df = pd.concat([
        ancient_df[['genetic_id', 'age']],
        modern_df[['genetic_id', 'age']]
    ], ignore_index=True)
    records = []
    missing_record_count = 0

    existing_fasta_paths = [f for f in fasta_data_filepaths if os.path.exists(f)]
    fasta_index = SeqIO.to_dict((rec for f in existing_fasta_paths for rec in SeqIO.parse(f, "fasta")))

    for _, row in combined_df.iterrows():
        genetic_id = row['genetic_id']
        age        = row['age']
        record     = fasta_index.get(genetic_id)

        if record is not None:
            new_record = SeqRecord(
                seq = record.seq,
                id  = str(genetic_id),
                description = str(age)
            )
            records.append(new_record)
        else:
            print(f"Could not find record for {genetic_id}")
            missing_record_count += 1

    # Write all records to a file
    mega_fasta_filename = "../data/generated/" + output_name
    count = SeqIO.write(records, mega_fasta_filename, "fasta")

    print(f"Successfully wrote {count} records to {mega_fasta_filename}")
    if (missing_record_count > 0):
        print(f"Missing {missing_record_count}")

    return

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

        balanced_df_a.drop(columns=['set'], inplace=True)
        balanced_df_b.drop(columns=['set'], inplace=True)

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

# create_mega_fasta(ancient_df_balanced, modern_df_balanced, "mega.fasta")

# Execute splits
anc_train, anc_val, anc_test = split_three_way(ancient_df_balanced, "Ancient")
mod_train, mod_val, mod_test = split_three_way(modern_df_balanced, "Modern")

create_mega_fasta(anc_train, mod_train, "mega_train.fasta")
create_mega_fasta(anc_val, mod_val,     "mega_val.fasta")
create_mega_fasta(anc_test, mod_test,   "mega_test.fasta")

output_dir = "../data/generated/training-metadata"
os.makedirs(output_dir, exist_ok=True)

datasets = {
    "anc_train.csv": anc_train,
    "anc_val.csv": anc_val,
    "anc_test.csv": anc_test,
    "mod_train.csv": mod_train,
    "mod_val.csv": mod_val,
    "mod_test.csv": mod_test
}

for filename, df in datasets.items():
    file_path = os.path.join(output_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")
