from Bio import Entrez
import time

Entrez.email = "_____@mymacewan.ca" # IMPORTANT - NEED EMAIL FOR NCBI

def search_genbank(search_term, retmax=100000):
    print(f"Searching GenBank with query: {search_term}")
    handle = Entrez.esearch(
        db="nucleotide",
        term=search_term,
        retmax=retmax,
        usehistory="y"
    )
    record = Entrez.read(handle)
    handle.close()
    count = int(record["Count"])
    print(f"Found {count} sequences")
    return record["WebEnv"], record["QueryKey"], count

def download_sequences_batch(webenv, query_key, count, batch_size=500, output_file="modern_mtdna.fasta"):
    print(f"Downloading {count} sequences in batches of {batch_size}")
    with open(output_file, "w") as out_handle:
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            print(f"Downloading records {start+1} to {end}")
            attempt = 0
            max_attempts = 3
            while attempt < max_attempts:
                try:
                    fetch_handle = Entrez.efetch(
                        db="nucleotide",
                        rettype="fasta",
                        retmode="text",
                        retstart=start,
                        retmax=batch_size,
                        webenv=webenv,
                        query_key=query_key
                    )
                    data = fetch_handle.read()
                    fetch_handle.close()
                    out_handle.write(data)
                    break  # Success
                except Exception as e:
                    attempt += 1
                    print(f"Error on attempt {attempt}: {e}")
                    if attempt < max_attempts:
                        print("Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        print(f"Failed to download batch {start}-{end}")
            time.sleep(0.5)
    
    print(f"Download complete. Saved as: {output_file}")

if __name__ == "__main__":
    # filter for modern human mitochondrial DNA sequences (1950 - 2025)
    search_term = (
        '"Homo sapiens"[Organism] AND '
        'mitochondrion[filter] AND '
        '16000:18000[Sequence Length] AND '
        '1950:2025[Publication Date]'
    )
    webenv, query_key, count = search_genbank(search_term, retmax=100000)
    output_file = "modern_mtdna_raw.fasta"
    download_sequences_batch(webenv, query_key, count, batch_size=500, output_file=output_file)