#!/usr/bin/env python3
"""
Finds the most recent ancient sample across the train, val, and
test feature CSVs by identifying the smallest absolute calendar year value
that is not 0 (since 0 is the modern GenBank sample).

Output:
    - Calendar year of the newest ancient sample (e.g. 1305 CE or -50 BCE)
    - Converted to years ago (e.g. 721 years ago)
    - Which split (train/val/test) it came from
    - The sample ID

Usage:
    python find_newest_ancient.py
    python find_newest_ancient.py --feature-dir path/to/features/
    python find_newest_ancient.py --verbose   # show top-10 newest ancient samples
"""

import os
import argparse
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, '..', 'data', 'generated', 'features')
PRESENT_YEAR = 2026

SPLITS = {
    'train': 'train_features.csv',
    'val':   'val_features.csv',
    'test':  'test_features.csv',
}


def find_newest_ancient(feature_dir, verbose=False, top_n=10):
    ""
    all_rows = []

    for split_name, filename in SPLITS.items():
        path = os.path.join(feature_dir, filename)
        if not os.path.exists(path):
            print(f'  WARNING: {path} not found — skipping.')
            continue

        df = pd.read_csv(path, dtype={'ID': str})
        if 'AGE' not in df.columns:
            print(f'  WARNING: No AGE column in {filename} — skipping.')
            continue

        df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')

        # Exclude modern sentinel (age == 0) and NaN
        ancient = df[df['AGE'].notna() & (df['AGE'] != 0)].copy()

        if ancient.empty:
            print(f'  WARNING: No ancient samples found in {filename}.')
            continue

        # "Newest" ancient = smallest absolute value of calendar year
        # A sample from 50 CE (age=50) has abs=50
        # A sample from 50 BCE (age=-50) has abs=50
        # Both are equally "recent" in calendar terms
        ancient['abs_age']   = ancient['AGE'].abs()
        ancient['years_ago'] = PRESENT_YEAR - ancient['AGE']
        ancient['split']     = split_name

        all_rows.append(ancient[['ID', 'AGE', 'abs_age', 'years_ago', 'split']])

    if not all_rows:
        print('ERROR: No ancient samples found in any split.')
        return None

    combined = pd.concat(all_rows, ignore_index=True)

    # Sort by years_ago ascending, smallest years_ago = most recent = newest ancient
    combined = combined.sort_values('years_ago', ascending=True).reset_index(drop=True)

    return combined


def format_calendar_year(cal_year):
    """Formats a calendar year as e.g. '1305 CE' or '277 BCE'."""
    if cal_year > 0:
        return f'{int(cal_year)} CE'
    else:
        return f'{int(abs(cal_year))} BCE'


def main():
    parser = argparse.ArgumentParser(
        description='Find the newest (most recent) ancient sample by AGE.'
    )
    parser.add_argument('--feature-dir', default=DATA_DIR,
                        help='Directory containing train/val/test feature CSVs')
    parser.add_argument('--verbose', action='store_true',
                        help='Print the top-10 newest ancient samples')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of newest samples to show with --verbose')
    args = parser.parse_args()

    print('=== Finding Newest Ancient Sample ===\n')

    combined = find_newest_ancient(args.feature_dir, args.verbose, args.top)
    if combined is None:
        return

    # The single newest ancient sample
    newest = combined.iloc[0]
    cal_year_str = format_calendar_year(newest['AGE'])
    years_ago    = int(newest['years_ago'])
    centuries    = years_ago / 100

    print(f'  Newest ancient sample')
    print(f'  {"─"*40}')
    print(f'  ID          : {newest["ID"]}')
    print(f'  Split       : {newest["split"]}')
    print(f'  Calendar age: {cal_year_str}  (raw AGE column value: {newest["AGE"]:.0f})')
    print(f'  Years ago   : {years_ago} years ago')
    print(f'  Centuries   : {centuries:.2f} centuries ago')
    print()
    print(f'  → To use this as your threshold, set:')
    print(f'    NEWEST_ANCIENT_AGE_YEARS = {years_ago}')
    print(f'    in classify.py, or run:')
    print(f'    python classify.py --threshold {years_ago}')

    if args.verbose:
        top_n = min(args.top, len(combined))
        print(f'\n  Top {top_n} newest ancient samples across all splits:')
        print(f'  {"─"*65}')
        header = f'  {"#":>3}  {"ID":<20}  {"Calendar Age":>13}  '
        header += f'{"Years Ago":>10}  {"Centuries":>9}  {"Split":<6}'
        print(header)
        print(f'  {"─"*65}')
        for i, row in combined.head(top_n).iterrows():
            cal_str = format_calendar_year(row['AGE'])
            ya      = int(row['years_ago'])
            cent    = ya / 100
            print(f'  {i+1:>3}  {row["ID"]:<20}  {cal_str:>13}  '
                  f'{ya:>10}  {cent:>9.2f}  {row["split"]:<6}')

    # Summary stats
    print(f'\n  Summary across all splits:')
    print(f'  Total ancient samples : {len(combined):,}')
    print(f'  Newest (min years ago): {int(combined["years_ago"].min())} years ago')
    print(f'  Oldest (max years ago): {int(combined["years_ago"].max()):,} years ago')
    print(f'  Median age            : {int(combined["years_ago"].median()):,} years ago')

    print('\n=== Done ===')


if __name__ == '__main__':
    main()
