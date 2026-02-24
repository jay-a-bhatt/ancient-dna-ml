import pandas as pd
import pycountry_convert as pc
from Bio import SeqIO

priority = {
    '.DG':    1, 
    '.AG.BY': 2, 
    '.SG':    2, 
    '.AG.SG': 3, 
    '.AG':    4, 
    '.HO':    5, 
    '.IM':    6, 
    '.TW':    6,
    '.AG.TW': 7
}

# Get a rank for different sample types since there may be
# mutiple types of sample types for each Master_ID
def get_rank(gid):
    sorted_suffixes = sorted(priority.keys(), key=len, reverse=True)
    
    for suffix in sorted_suffixes:
        if suffix in gid:
            return priority[suffix]
    return 8 # Unknown samples type

# Gets the continent name for a county.
def get_continent(country_name):
    try:
        # Convert country name to 2-letter country code (ISO)
        country_code = pc.country_name_to_country_alpha2(country_name)
        # Convert country code to continent code
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        # Convert continent code to full name
        continent_names = {
            'AF': 'Africa', 'AS': 'Asia', 'EU': 'Europe', 
            'NA': 'North America', 'SA': 'South America', 
            'OC': 'Oceania', 'AN': 'Antarctica'
        }
        return continent_names.get(continent_code)
    except:
        return None

def get_amtDB_cols(file_path):
    amtDB_cols_to_keep = ['identifier', 'year_from', 'year_to', 'mt_hg', 'continent']

    try:
        df = pd.read_csv(file_path, usecols = amtDB_cols_to_keep)
        df['age'] = ((df['year_from'] + df['year_to']) / 2).round().astype(int)
        df = df.drop(columns=['year_from', 'year_to'])
        # Drop Rows with no haplogroup
        df = df.dropna(subset=['mt_hg'])
        df.rename(columns={df.columns[0]: 'Master_ID'}, inplace=True)
        df['Genetic_ID'] = df['Master_ID']
        new_order = ['Genetic_ID', 'Master_ID', 'continent', 'mt_hg', 'age']
        df_final = df[new_order]
        return df_final

    except FileNotFoundError:
        print("File not found.")

    except ValueError:
        print("One of the columns was not found in the CSV.")

def get_AADR_cols(metadata_filepath, fasta_filepath):
    ids = [record.id for record in SeqIO.parse(fasta_filepath, "fasta")]
    print(f"AADR fasta ID count: {len(ids)}")
    df_fasta_ids = pd.DataFrame(ids, columns=['Genetic_ID'])

    # Keep Genetic ID, age, country.
    df_meta = pd.read_excel(metadata_filepath, na_values=["n/a (<2x)", ".."], usecols=[0,1,9,15,31])
    df_meta.rename(columns={df_meta.columns[0]: 'Genetic_ID'}, inplace=True)
    df_meta.rename(columns={df_meta.columns[1]: 'Master_ID'}, inplace=True)
    df_meta.rename(columns={df_meta.columns[2]: 'age'}, inplace=True)
    df_meta.rename(columns={df_meta.columns[4]: 'mt_hg'}, inplace=True)

    # Convert ages to calendar year
    df_meta['age'] = pd.to_numeric(df_meta['age'], errors='coerce')
    df_meta['age'] = 1950 - df_meta['age']

    df_meta_filtered = pd.merge(df_fasta_ids, df_meta, left_on='Genetic_ID', right_on=df_meta.columns[0])
    df_final = df_meta_filtered.dropna(subset=[df_meta_filtered.columns[4]])

    # Assign a rank to each sample
    df_final['rank'] = df_final['Genetic_ID'].apply(get_rank)
    # Sort samples by rank
    df_final = df_final.sort_values(by=['Master_ID', 'rank'], ascending=True)
    # Keep the best sample type for each Master_ID
    df_final = df_final.drop_duplicates(subset=['Master_ID'], keep='first')

    # Create continent column using Political Entity (country) column
    df_final['continent'] = df_final['Political Entity'].apply(get_continent)
    # Reorder columns
    new_order = ['Genetic_ID', 'Master_ID', 'continent', 'mt_hg', 'age']
    df_final = df_final[new_order]

    return df_final

amtDB_df = get_amtDB_cols("../data/metadata/amtdb_metadata.csv")
aadr_df  = get_AADR_cols("../data/metadata/v62.0_1240k_public.xlsx", "../data/aadr.fasta")

# If a Master_ID exists in both, keep='first' keeps the amtDB row
combined_df = pd.concat([amtDB_df, aadr_df], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset=['Master_ID'], keep='first')

# drop rows with no continent
combined_df = combined_df.dropna(subset=['continent'])

# Save to CSV
output_filename = "ancient_dna_metadata.csv"
combined_df.to_csv(output_filename, index=False)
