import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def load_bootstrap_data(data_path, metrics, responders=None):
    all_data = {}
    
    for filename in [f for f in os.listdir(data_path) if f.endswith('_bootstrap.csv')]:
        parts = filename.replace('_bootstrap.csv', '').split(' x ')
        if len(parts) != 2: continue
        mbon, responder = parts
        if responders and responder not in responders: continue
        if responder not in all_data:
            all_data[responder] = {m: {} for m in metrics}
        
        df = pd.read_csv(os.path.join(data_path, filename))
        for (intensity, metric), group in df.groupby(['Light_Intensity', 'Metric']):
            if intensity == "Full" and metric in metrics:
                all_data[responder][metric][mbon] = {
                    'bootstrap': group['Bootstrap'].values,
                    'effect': group['Hedges_g'].iloc[0],
                    'ci_low': group['CI_low'].iloc[0],
                    'ci_high': group['CI_high'].iloc[0]
                }
    return all_data

def get_effect(data, responder, metric, mbon):
    d = data[responder][metric].get(mbon, {})
    return d.get('effect'), d.get('ci_low'), d.get('ci_high')

def get_bootstrap(data, responder, metric, mbon):
    return data[responder][metric].get(mbon, {}).get('bootstrap', np.zeros(5000))

def get_mbons(data, responder):
    mbons = set()
    for metric in data[responder]:
        mbons.update(data[responder][metric].keys())
    return list(mbons)

def get_mbons_by_number(data, responder, lobelocation):
    mbons = set()
    for metric in data[responder]:
        mbons.update(data[responder][metric].keys())
    
    if lobelocation is None or lobelocation.empty:
        return sorted(list(mbons))
    
    def sort_key(mbon):
        info = lobelocation[lobelocation['MBON'] == mbon]
        if len(info) > 0:
            match = re.search(r'(\d+)', str(info['MBON_number'].values[0]).strip())
            if match: return int(match.group(1))
        return 999
    return sorted(list(mbons), key=sort_key)

def find_number(df, lookup, col):
    vals = sorted({str(row[col]).strip() for _, row in df.iterrows()
                   if lookup in [x.strip() for x in str(row['MBON names']).split(',')]})
    pre = re.match(r'\D*', vals[0]).group() if vals else ''
    if len(vals) > 1 and pre and all(v.startswith(pre) and v[len(pre):].isdigit() for v in vals):
        return pre + ', '.join(v[len(pre):] for v in vals)
    return ', '.join(vals)

def create_mbon_only_labels(mbons, lobelocation):
    if lobelocation is None or lobelocation.empty: return list(mbons)
    # First pass: get compressed MBON number for each mbon
    raw_labels = []
    for mbon in mbons:
        info = lobelocation[lobelocation['MBON'] == mbon]
        if len(info) > 0:
            raw_number = str(info['MBON_number'].values[0])
            label = raw_number
            raw_labels.append(label if label and label != 'nan' else mbon)
        else:
            raw_labels.append(mbon)
    # Second pass: add _1, _2 suffixes only for duplicates
    counts = Counter(raw_labels)
    seen = {}
    labels = []
    for label in raw_labels:
        if counts[label] > 1:
            seen[label] = seen.get(label, 0) + 1
            labels.append(f"{label}_{seen[label]}")
        else:
            labels.append(label)
    return labels

def create_labels(mbons, lobelocation, sep='\n'):
    if lobelocation is None or lobelocation.empty: return list(mbons)
    labels = []
    for mbon in mbons:
        info = lobelocation[lobelocation['MBON'] == mbon]
        if len(info) > 0:
            #parts = [str(info['MBON_number'].values[0]), str(info['Lobe_location'].values[0]), mbon]
            raw_number = str(info['MBON_number'].values[0])
            parts = [raw_number]
            labels.append(sep.join([p for p in parts if p and p != 'nan']) or mbon)
        else:
            labels.append(mbon)
    return labels

def _spiralize(fill, m, n):
    i, j, k = 0, 0, 0
    arr = np.zeros((m, n))
    while m > 0 and k < len(fill):
        jj, ii = j, i
        for j in range(j, n):
            if k >= len(fill): break
            arr[i,j] = fill[k]; k += 1
        for i in range(ii+1, m):
            if k >= len(fill): break
            arr[i,j] = fill[k]; k += 1
        for j in range(n-2, jj-1, -1):
            if k >= len(fill): break
            arr[i,j] = fill[k]; k += 1
        for i in range(m-2, ii, -1):
            if k >= len(fill): break
            arr[i,j] = fill[k]; k += 1
        m -= 1; n -= 1; j += 1
    return arr

def _sample_bootstrap(bootstrap, m, n, chop_tail=2.5):
    bs = sorted(bootstrap)
    chop = int(np.ceil(len(bs) * chop_tail / 100))
    bs = bs[chop:len(bs)-chop]
    ranks = np.linspace(0, len(bs), m*n, dtype=int)
    ranks[0] = 1
    if np.sum(np.array(bs) > 0) < len(bs)//2: bs = bs[::-1]
    return [bs[r-1] for r in ranks]

def _get_text_color(spirals, row_idx, col_idx, n, cmap, vmin, vmax):
    cy, cx = row_idx * n + n // 2, col_idx * n + n // 2
    center_val = np.mean(spirals[cy-1:cy+2, cx-1:cx+2])
    norm_val = np.clip((center_val - vmin) / (vmax - vmin), 0, 1) if vmax != vmin else 0.5
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    rgba = cmap_obj(norm_val)
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return 'white' if luminance < 0.5 else 'black'

def build_vortex_df(data, responder, metrics, lobelocation, n=11, chop_tail=2.5):
    mbons = get_mbons_by_number(data, responder, lobelocation)
    n_rows = len(mbons)
    n_cols = len(metrics)
    
    spirals = np.zeros((n_rows * n, n_cols * n))
    mean_vals = np.zeros((n_rows, n_cols))
    
    for i, mbon in enumerate(mbons):
        for j, metric in enumerate(metrics):
            bootstrap = get_bootstrap(data, responder, metric, mbon)
            sampled = _sample_bootstrap(bootstrap, n, n, chop_tail)
            spiral = _spiralize(sampled, n, n)
            spirals[i*n:(i+1)*n, j*n:(j+1)*n] = spiral
            effect, _, _ = get_effect(data, responder, metric, mbon)
            mean_vals[i, j] = effect if effect is not None else np.nan
    
    return spirals, mean_vals, mbons

def vortex_map(spirals, mean_vals, n=11, cmap='coolwarm', vmin=-1, vmax=1, 
               figsize=None, annot=True, annot_fontsize=9, annot_fmt='.2f'):
    n_rows, n_cols = mean_vals.shape
    
    fig, ax = plt.subplots(figsize=figsize or (n_cols * 1.5, n_rows * 0.6))
    ax.imshow(spirals, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Remove spines (outer bounding box)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    if annot:
        for i in range(n_rows):
            for j in range(n_cols):
                color = _get_text_color(spirals, i, j, n, cmap, vmin, vmax)
                ax.text(j*n + (n-1)/2, i*n + (n-1)/2, f'{mean_vals[i,j]:{annot_fmt}}',
                        ha='center', va='center', fontsize=annot_fontsize, color=color)
    
    ax.set_xticks([j*n + (n-1)/2 for j in range(n_cols)])
    ax.set_yticks([i*n + (n-1)/2 for i in range(n_rows)])
    
    return fig, ax
