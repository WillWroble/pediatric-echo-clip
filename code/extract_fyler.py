import glob, re, csv
from collections import defaultdict
import openpyxl

XLSX_DIR = '/lab-share/Cardio-Mayourian-e2/Public/Echo_Labels/Full_Qualitative_XLSX'
OUT_DIR = '/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip'
IGNORE_PATH = '/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/ignore_fyler_lines.txt'
CODE_RE = re.compile(r'^(.+?)\s*\[(\d+)\]\s*$')

# Load ignore list
ignore_lines = set()
with open(IGNORE_PATH) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            ignore_lines.add(line)
print(f'Loaded {len(ignore_lines)} ignore lines')

line_counts = defaultdict(lambda: defaultdict(int))  # code -> {desc: count}
studies = {}

for f in sorted(glob.glob(f'{XLSX_DIR}/AIecho_qualitative*.xlsx')):
    print(f'Reading {f}...')
    wb = openpyxl.load_workbook(f, read_only=True)
    for row in wb.active.iter_rows(min_row=4):
        sid = row[6].value
        raw = row[7].value
        if sid is None or raw is None:
            continue
        sid = int(sid)
        raw = raw.strip()
        m = CODE_RE.match(raw)
        if not m:
            print(f'  WARN: no code in {repr(raw)}')
            continue
        desc = re.sub(r'\s*\[\d+\]\s*$', '', raw)
        code = m.group(2)

        if desc in ignore_lines:
            continue

        line_counts[code][desc] += 1

        if sid not in studies:
            studies[sid] = {
                'mrn': row[0].value,
                'dob': str(row[1].value)[:10] if row[1].value else '',
                'gender': row[2].value or '',
                'age': row[3].value or '',
                'event_date': str(row[4].value)[:10] if row[4].value else '',
                'location': row[5].value or '',
                'codes': set()
            }
        studies[sid]['codes'].add(code)
    wb.close()

all_codes = sorted({c for s in studies.values() for c in s['codes']}, key=int)
total_lines = sum(len(descs) for descs in line_counts.values())
print(f'\n{len(studies)} studies, {len(all_codes)} unique Fyler codes, {total_lines} unique lines')

with open(f'{OUT_DIR}/fyler_lines_v2.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['line', 'fyler_code', 'count'])
    for code in sorted(line_counts.keys(), key=int):
        for desc, count in sorted(line_counts[code].items(), key=lambda x: -x[1]):
            w.writerow([desc, code, count])

with open(f'{OUT_DIR}/fyler_labels_v2.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['sid', 'mrn', 'dob', 'gender', 'age', 'event_date', 'location'] + [f'fyler_{c}' for c in all_codes])
    for sid in sorted(studies.keys()):
        s = studies[sid]
        row = [sid, s['mrn'], s['dob'], s['gender'], s['age'], s['event_date'], s['location']]
        row += [1 if c in s['codes'] else 0 for c in all_codes]
        w.writerow(row)

print(f'Wrote {OUT_DIR}/fyler_lines_v2.csv and {OUT_DIR}/fyler_labels_v2.csv')
