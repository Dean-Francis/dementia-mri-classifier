from pathlib import Path

data_root = r'C:\Users\user\Desktop\Convolutional_Neural_Netowork\Dr. Firuz Kamalov\images\disc1'

cn_count = 0
ad_count = 0

for subject_dir in sorted(Path(data_root).iterdir()):
    if not subject_dir.is_dir() or 'OAS1_' not in subject_dir.name:
        continue
    
    txt_files = list(subject_dir.glob('*_MR1.txt'))
    if not txt_files:
        continue
    
    try:
        with open(txt_files[0], 'r') as f:
            for line in f:
                if 'CDR:' in line:
                    cdr_str = line.split(':')[1].strip()
                    if cdr_str and cdr_str != '':
                        cdr = float(cdr_str)
                        if cdr == 0:
                            cn_count += 1
                        else:
                            ad_count += 1
                        print(f"{subject_dir.name}: CDR={cdr}")
                    break
    except:
        pass

print(f"\nTotal CN: {cn_count}")
print(f"Total AD: {ad_count}")