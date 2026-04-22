import glob, re, csv
import openpyxl

XLSX_DIR = '/lab-share/Cardio-Mayourian-e2/Public/Echo_Labels/Full_Qualitative_XLSX'
OUT_DIR = '/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip'
CODE_RE = re.compile(r'^(.+?)\s*\[(\d+)\]\s*$')

lines_seen = {}  # raw_string -> code
studies = {}     # sid -> {meta, codes}

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
        code = m.group(2)
        lines_seen[raw] = code
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
print(f'\n{len(studies)} studies, {len(all_codes)} unique Fyler codes, {len(lines_seen)} unique lines')

with open(f'{OUT_DIR}/fyler_lines.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['line', 'fyler_code'])
    for line, code in sorted(lines_seen.items(), key=lambda x: int(x[1])):
        w.writerow([line, code])

with open(f'{OUT_DIR}/fyler_labels.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['sid', 'mrn', 'dob', 'gender', 'age', 'event_date', 'location'] + [f'fyler_{c}' for c in all_codes])
    for sid in sorted(studies.keys()):
        s = studies[sid]
        row = [sid, s['mrn'], s['dob'], s['gender'], s['age'], s['event_date'], s['location']]
        row += [1 if c in s['codes'] else 0 for c in all_codes]
        w.writerow(row)

print(f'Wrote {OUT_DIR}/fyler_lines.csv and {OUT_DIR}/fyler_labels.csv')
