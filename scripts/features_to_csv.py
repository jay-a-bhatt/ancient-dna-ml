#!/usr/bin/env python3
"""
Strategy:
  - DATABASE : training set only
  - QUERY    : ALL samples (train + val + test), one at a time
  - Self-matches are excluded when computing the NRC weighted average age

Output (one CSV per split):
    /data/generated/features/train_features.csv
    /data/generated/features/val_features.csv
    /data/generated/features/test_features.csv

Columns:
    ID, SEQUENCE, NRC_AVERAGE_AGE, CG_CONTENT, N_CONTENT, RELATIVE_SIZE

Usage:
    python features_to_csv.py
    python features_to_csv.py --falcon ./FALCON --top 50 --threads 12 --ref-size 17000
    python features_to_csv.py --unweighted   # simple average instead of weighted
"""

import os
import re
import argparse
import subprocess
import tempfile
import shutil

import pandas as pd

# Config

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, '..', 'data', 'generated')
TRAIN_FASTA  = os.path.join(DATA_DIR, 'mega_train.fasta')
VAL_FASTA    = os.path.join(DATA_DIR, 'mega_val.fasta')
TEST_FASTA   = os.path.join(DATA_DIR, 'mega_test.fasta')
OUTPUT_DIR   = os.path.join(DATA_DIR, 'features')

FALCON_MODELS = ['6:1:0:0/0', '11:10:0:0/0', '13:200:1:3/1']

# Parsing

def parse_fasta(filepath):
    """
    Parses a multi-FASTA file with headers of the form
        >ID AGE    e.x. >I17261.AG -274
    Returns (sample_id, age_float_or_None, sequence) as list
    """
    with open(filepath, 'r') as f:
        content = f.read().upper()

    pattern = re.compile(r'>(.*?)\n([\s\S]*?)(?=\n>|\Z)', re.DOTALL)
    samples = []
    for header, seq in pattern.findall(content):
        seq      = seq.replace('\n', '')
        parts    = header.strip().split(' ', 1)   # split on first space only
        sample_id = parts[0]
        try:
            age = float(parts[1]) if len(parts) > 1 else None
        except ValueError:
            age = None
        samples.append((sample_id, age, seq))
    return samples

# FALCON

def run_falcon_for_sample(falcon_bin, sample_id, age, seq,
                           train_fasta, tops_dir, top_n, threads):
    """
    Writes a single-sequence temp FASTA, runs FALCON against the training
    database, parses the top file, then deletes both temp files.
    Returns a list of [rank, length, similarity, db_id, db_age].
    """
    # Write temp query FASTA
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa',
                                     dir=tops_dir, delete=False) as tmp:
        tmp.write(f'>{sample_id} {age}\n{seq}\n')
        temp_fasta_path = tmp.name

    top_file = os.path.join(tops_dir, f'{sample_id}_top.txt')

    cmd = [falcon_bin]
    for m in FALCON_MODELS:
        cmd += ['-m', m]
    cmd += ['-g', '0.85', '-F',
            '-t', str(top_n),
            '-n', str(threads),
            '-x', top_file,
            temp_fasta_path, train_fasta]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'    FALCON stderr: {result.stderr.strip()}')

    os.unlink(temp_fasta_path)

    results = []
    if not os.path.exists(top_file):
        print(f'    FALCON did not produce a top file for {sample_id}')
        return results

    with open(top_file, 'r') as f:
        raw = f.read()

    if not raw.strip():
        print(f'    FALCON top file is empty for {sample_id}')
        os.unlink(top_file)
        return results

    for line in raw.splitlines():
        parts = line.strip().split('\t')
        if len(parts) < 4:
            continue
        db_header = parts[3]

        # Header format from FALCON output might vary (tries space first, then underscore)
        if ' ' in db_header:
            db_parts = db_header.split(' ', 1)
            db_id    = db_parts[0]
            try:
                db_age = float(db_parts[1]) if len(db_parts) > 1 else None
            except ValueError:
                db_age = None
        else:
            # Fallback: underscore-delimited (e.g. ID_age)
            db_parts = db_header.rsplit('_', 1)
            db_id    = db_parts[0]
            try:
                db_age = float(db_parts[1]) if len(db_parts) > 1 else None
            except ValueError:
                db_age = None

        results.append([
            int(parts[0]),    # rank
            int(parts[1]),    # length
            float(parts[2]),  # similarity
            db_id,
            db_age,
        ])

    os.unlink(top_file)
    return results


def compute_nrc_age(top_list, sample_id, weighted=True):
    """
    Computes average age from FALCON top matches, excluding self-matches.
    weighted = True  -> similarity-weighted average
    weighted = False -> simple average (same as the 2025 paper)
    Returns float or None if no valid matches.
    """
    non_self = [item for item in top_list if item[3] != sample_id and item[4] is not None]

    if not non_self:
        return None

    if weighted:
        norm_val = sum(item[2] for item in non_self)
        if norm_val == 0:
            return None
        return sum(item[2] * item[4] for item in non_self) / norm_val
    else:
        return sum(item[4] for item in non_self) / len(non_self)

# Quantitative features

def compute_quantitative(seq, ref_size):
    seq     = seq.upper()
    seq_len = len(seq)
    if seq_len == 0:
        return None, None, None
    cg_content    = (seq.count('C') + seq.count('G')) / seq_len
    n_content     = seq.count('N') / seq_len
    relative_size = seq_len / ref_size
    return cg_content, n_content, relative_size


# Main pipeline

def extract_features(all_samples, train_fasta, falcon_bin, top_n, threads, ref_size, weighted):
    """
    Runs FALCON for every sample and computes all features.
    Returns a dict keyed by sample_id -> feature row.
    Deletes the intermediate files.
    """
    tops_dir = tempfile.mkdtemp(prefix='falcon_tops_')

    features = {}
    total    = len(all_samples)

    try:
        for i, (sample_id, age, seq) in enumerate(all_samples, 1):
            print(f'  [{i}/{total}] {sample_id} (age={age})')

            top_list    = run_falcon_for_sample(
                falcon_bin, sample_id, age, seq,
                train_fasta, tops_dir, top_n, threads
            )
            nrc_avg_age = compute_nrc_age(top_list, sample_id, weighted)
            cg, n, rel  = compute_quantitative(seq, ref_size)

            if nrc_avg_age is None:
                print(f'    WARNING: no valid NRC matches for {sample_id}, skipping.')
                continue

            features[sample_id] = {
                'ID':              sample_id,
                'SEQUENCE':        seq,
                'NRC_AVERAGE_AGE': nrc_avg_age,
                'CG_CONTENT':      cg,
                'N_CONTENT':       n,
                'RELATIVE_SIZE':   rel,
                '_age':            age,   # kept internally for reference, not saved
            }
    finally:
        shutil.rmtree(tops_dir, ignore_errors=True)

    return features


def save_split(split_samples, features, split_name, output_dir):
    # Builds + saves the feature csv for one split
    rows = []
    for sample_id, age, seq in split_samples:
        if sample_id not in features:
            print(f'  WARNING: {sample_id} missing from features, skipping.')
            continue
        f = features[sample_id]
        rows.append({
            'ID':              f['ID'],
            'SEQUENCE':        f['SEQUENCE'],
            'NRC_AVERAGE_AGE': f['NRC_AVERAGE_AGE'],
            'CG_CONTENT':      f['CG_CONTENT'],
            'N_CONTENT':       f['N_CONTENT'],
            'RELATIVE_SIZE':   f['RELATIVE_SIZE'],
        })

    out_path = os.path.join(output_dir, f'{split_name}_features.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'  Saved -> {out_path}  ({len(rows)} samples)')

# Main

def main():
    parser = argparse.ArgumentParser(
        description='Extract FALCON + quantitative features into ML-ready CSVs.'
    )
    parser.add_argument('--train',       default=TRAIN_FASTA)
    parser.add_argument('--val',         default=VAL_FASTA)
    parser.add_argument('--test',        default=TEST_FASTA)
    parser.add_argument('--falcon',      default='./FALCON',
                        help='Path to FALCON binary')
    parser.add_argument('--top',         type=int, default=50,
                        help='Top N matches (default: 50)')
    parser.add_argument('--threads',     type=int, default=12,
                        help='Threads (default: 12)')
    parser.add_argument('--ref-size',    type=int, default=17000,
                        help='Reference genome size in bp (default: 17000)')
    parser.add_argument('--outdir',      default=OUTPUT_DIR)
    parser.add_argument('--unweighted',  action='store_true',
                        help='Simple average for NRC age (paper text). '
                             'Default: similarity-weighted (source code).')
    args    = parser.parse_args()
    weighted = not args.unweighted

    os.makedirs(args.outdir, exist_ok=True)

    print('\nParsing FASTA files')
    train_samples = parse_fasta(args.train)
    val_samples   = parse_fasta(args.val)
    test_samples  = parse_fasta(args.test)
    all_samples   = train_samples + val_samples + test_samples
    print(f'Train: {len(train_samples)} | Val: {len(val_samples)} | '
          f'Test: {len(test_samples)} | Total: {len(all_samples)}')

    print(f'\nRunning FALCON + extracting features (weighted={weighted})')
    features = extract_features(
        all_samples, args.train, args.falcon,
        args.top, args.threads, args.ref_size, weighted
    )

    print('\nSaving per-split CSVs')
    save_split(train_samples, features, 'train', args.outdir)
    save_split(val_samples,   features, 'val',   args.outdir)
    save_split(test_samples,  features, 'test',  args.outdir)

    print('\nDone')
    print(f'Output: {args.outdir}/')
    print('  train_features.csv')
    print('  val_features.csv')
    print('  test_features.csv')
    print('\nColumns: ID | SEQUENCE | NRC_AVERAGE_AGE | CG_CONTENT | N_CONTENT | RELATIVE_SIZE')
    print('\nTo create your binary label:')
    print('  df["label"] = (df["AGE"] > threshold_years).astype(int)')


if __name__ == '__main__':
    main()
