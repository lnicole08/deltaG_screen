import re
import textwrap
import pandas as pd


def find_number(df, lookup, col):
    vals = sorted({str(row[col]).strip() for _, row in df.iterrows()
                   if lookup in [x.strip() for x in str(row['MBON names']).split(',')]})
    pre = re.match(r'\D*', vals[0]).group() if vals else ''
    if len(vals) > 1 and pre and all(v.startswith(pre) and v[len(pre):].isdigit() for v in vals):
        return pre + ', '.join(v[len(pre):] for v in vals)
    return ', '.join(vals)

def generate_lobelocation(mbon_list, csv_path):
    csvfile = pd.read_csv(csv_path).astype('string')
    rows = [{'MBON': m,
             'Lobe_location': find_number(csvfile, m, "Lobe"),
             'MBON number': find_number(csvfile, m, "MBON number"),
             'Neurotransmitter': find_number(csvfile, m, "Neurotransmitter")}
            for m in mbon_list]
    lobelocation = pd.DataFrame(rows)
    lobelocation['MBON_number'] = lobelocation['MBON number']
    return lobelocation


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


def parse_lobes(lobe_str):
    if pd.isna(lobe_str) or lobe_str == '':
        return ['Unknown']
    lobe_str_lower = lobe_str.lower()
    lobes = []
    if 'calyx' in lobe_str_lower: lobes.append('Calyx')
    if "α'" in lobe_str or "a'" in lobe_str_lower: lobes.append("α'")
    if "β'" in lobe_str or "b'" in lobe_str_lower: lobes.append("β'")
    if any(c in lobe_str_lower for c in ['a1', 'a2', 'a3', 'α1', 'α2', 'α3']) and "α'" not in lobes: lobes.append('α')
    if any(c in lobe_str_lower for c in ['b1', 'b2', 'b3', 'β1', 'β2', 'β3']) and "β'" not in lobes: lobes.append('β')
    if any(c in lobe_str_lower for c in ['y1', 'y2', 'y3', 'y4', 'y5', 'γ1', 'γ2', 'γ3', 'γ4', 'γ5']): lobes.append('γ')
    return lobes if lobes else ['Unknown']


def parse_nt(nt_str):
    if pd.isna(nt_str) or nt_str == '':
        return ['Unknown']
    nts = [x.strip() for x in nt_str.split(',')]
    return nts if nts else ['Unknown']
