import pandas as pd
import matplotlib.pyplot as plt

from Bio import SeqIO

# Path to your file
file_path = "../data/modern_mtdna_raw.fasta"

# Initialize a list to store our data
records_data = []
unverified_count = 0
unknown_haplogroup_count = 0
record_count = 0

def plot_haplotype_counts(df, column_name, top_n=20):
    """
    Takes a DataFrame and plots a bar chart of the counts of values in a column.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the results.
    column_name (str): The name of the column to count.
    top_n (int): The number of top haplotypes to display (default 20).
    """
    # Calculate counts and take the top N
    counts = df[column_name].value_counts().head(top_n)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    
    # Formatting
    plt.title(f'Top {top_n} Haplotypes by Frequency ($N={len(df)}$)', fontsize=14)
    plt.xlabel('Haplotype', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def categorize_origin(haplo):
    h = str(haplo).upper()
    
    # AFRICA (The Root)
    if h.startswith('L'):
        return "Africa"
    
    # AMERICAS (Indigenous Sub-clades)
    # North America specific markers
    if h.startswith(('X2A', 'A2A', 'A2B', 'A2G')):
        return "North America"

    # South America (D1, C1b/c, and B2 are the dominant southern signatures)
    if h.startswith(('D1', 'C1B', 'C1C', 'D4H3A', 'B2I')):
        return "South America"

    # General Founding Lineages (Default to South if not specific, 
    # as South America has higher indigenous lineage retention in GenBank)
    if h.startswith(('A2', 'B2', 'C1')):
        return "South America"

    # EUROPE
    # These lineages (HV, JT, UK, IWX) define the European & Middle Eastern pool
    if h.startswith(('H', 'V', 'J', 'T', 'U', 'K', 'I', 'W', 'X')):
        return "Europe"

    # ASIA
    # These are the major Asian branches (M and N descendants)
    if h.startswith(('A', 'B', 'C', 'D', 'G', 'F', 'M', 'N', 'Y', 'Z')):
        return "Asia"

    # OCEANIA
    if h.startswith(('P', 'Q', 'S')):
        return "Oceania"

    return "Other/Unknown"

# Parse the FASTA file
for record in SeqIO.parse(file_path, "fasta"):
    record_count += 1
    header = record.description

    # Count UNVERIFIED
    is_unverified = "UNVERIFIED" in header.upper()
    if is_unverified:
        unverified_count += 1
        continue
    
    # Extract Haplogroup
    # We split the string by the anchor and take the first "word" after it
    anchor = " haplogroup "
    if anchor in header:
        # header.split(anchor)[1] gives everything after " haplogroup "
        # .split()[0] grabs only the first word (the haplogroup ID)
        haplogroup = header.split(anchor)[1].split()[0]
    else:
        haplogroup = "Unknown"
        unknown_haplogroup_count += 1
        continue

    records_data.append({
        "genetic_id": record.id,
        "master_id": record.id,
        "mt_hg": haplogroup,
        "age": 0,
    })

# Convert to DataFrame
df = pd.DataFrame(records_data)
df['continent'] = df['mt_hg'].apply(categorize_origin)

print(f"Number of 'Other/Unknown' records to be removed: {(df['continent'] == "Other/Unknown").sum()}")
df = df[df['continent'] != "Other/Unknown"].copy()

# Save to CSV
df.to_csv("modern_mtDNA_metadata.csv", index=False)

# Output results
print(f"Total records: {record_count}")
print(f"Total 'UNVERIFIED' records found: {unverified_count}")
print(f"Total unknown haplogroup records found: {unknown_haplogroup_count}")
print(f"Total sequences parsed: {len(df)}")
print("\nPreview of DataFrame:")
print(df.head())
