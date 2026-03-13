#!/usr/bin/env python3
"""
Adds an AGE column to existing feature CSVs by reading the age directly
from the corresponding FASTA file headers.

FASTA header format: >ID AGE    e.g. >I17261.AG -274

Matches each row in the CSV to its FASTA header by ID, then inserts
the AGE column immediately after the ID column.

File structure:
    /data/generated/mega_train.fasta  ->  /data/generated/features/train_features.csv
    /data/generated/mega_val.fasta    ->  /data/generated/features/val_features.csv
    /data/generated/mega_test.fasta   ->  /data/generated/features/test_features.csv

Modifies the CSVs in-place. Prints a summary of any IDs that could not
be matched between the CSV and the FASTA.
"""

import os
import re
import argparse

import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, '..', 'data', 'generated')
FEATURE_DIR = os.path.join(DATA_DIR, 'features')

SPLITS = [
    ('mega_train.fasta', 'train_features.csv'),
    ('mega_val.fasta',   'val_features.csv'),
    ('mega_test.fasta',  'test_features.csv'),
]


def parse_fasta_ages(fasta_path):
    """
    Reads a FASTA file and returns a dict of {ID: age_float}.
    Header format: >ID AGE   e.g. >I17261.AG -274
    """
    id_to_age = {}
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                continue
            header = line[1:]                       # strip leading >
            parts  = header.split(' ', 1)           # split on first space only
            sample_id = parts[0]
            try:
                age = float(parts[1]) if len(parts) > 1 else None
            except ValueError:
                age = None
            id_to_age[sample_id] = age
    return id_to_age


def add_age_to_csv(csv_path, fasta_path):
    """
    Loads the CSV, looks up each ID in the FASTA age map,
    inserts AGE as the second column (after ID), and saves in-place.
    """
    print(f'\n  CSV:   {csv_path}')
    print(f'  FASTA: {fasta_path}')

    df        = pd.read_csv(csv_path, dtype={'ID': str})
    id_to_age = parse_fasta_ages(fasta_path)

    if 'AGE' in df.columns:
        print('  AGE column already exists — overwriting with fresh values from FASTA.')
        df = df.drop(columns=['AGE'])

    df.insert(1, 'AGE', df['ID'].map(id_to_age))

    # Report IDs that failed to match
    missing = df[df['AGE'].isna()]['ID'].tolist()
    if missing:
        print(f'  WARNING: {len(missing)} IDs had no matching FASTA header:')
        for m in missing[:10]:
            print(f'    {m}')
        if len(missing) > 10:
            print(f'    ... and {len(missing) - 10} more')
    else:
        print(f'  All {len(df)} IDs matched successfully.')

    df.to_csv(csv_path, index=False)
    print(f'  Saved  -> {csv_path}')
    print(f'  Columns: {list(df.columns)}')
    print(f'  AGE range: {df["AGE"].min()} to {df["AGE"].max()} years')


def main():
    parser = argparse.ArgumentParser(
        description='Add AGE column to feature CSVs from FASTA headers.'
    )
    parser.add_argument('--feature-dir', default=FEATURE_DIR,
                        help='Directory containing the feature CSVs')
    parser.add_argument('--fasta-dir',   default=DATA_DIR,
                        help='Directory containing the FASTA files')
    parser.add_argument('--splits', nargs='+',
                        metavar='FASTA:CSV',
                        default=None,
                        help='Optional custom split pairs as FASTA_filename:CSV_filename. '
                             'e.g. mega_train.fasta:train_features.csv')
    args = parser.parse_args()

    splits = SPLITS
    if args.splits:
        splits = []
        for pair in args.splits:
            fasta_name, csv_name = pair.split(':')
            splits.append((fasta_name, csv_name))

    print('=== Adding AGE column to feature CSVs ===')
    for fasta_name, csv_name in splits:
        fasta_path = os.path.join(args.fasta_dir,   fasta_name)
        csv_path   = os.path.join(args.feature_dir, csv_name)

        if not os.path.exists(fasta_path):
            print(f'\n  ERROR: FASTA not found: {fasta_path}')
            continue
        if not os.path.exists(csv_path):
            print(f'\n  ERROR: CSV not found: {csv_path}')
            continue

        add_age_to_csv(csv_path, fasta_path)

    print('\n=== Done ===')


if __name__ == '__main__':
    main()
