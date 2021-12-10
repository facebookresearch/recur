import numpy as np
import urllib.request

############################  OEIS   ############################

# patterns:
# nice: an exceptionally nice sequence
# formula: FORMULA
# easy: very easy to produce terms of sequence

def read_url(address):
    try:
        with urllib.request.urlopen(address) as uf:
            text = str(uf.read())
    except: 
        with urllib.request.urlopen(address) as uf:
            text = str(uf.read())
    return text

def check_patterns(identifier, patterns=[], exclude=["prime"]):
    text = read_url("https://oeis.org/"+identifier)
    title = text[:text.find('OFFSET')]
    
    length = len(text)
    
    good = []
    bad  = []
    for pattern in patterns:
        good.append(pattern in text)
    for pattern in exclude:
        bad.append(pattern in title)
    valid = all(good) and not any(bad)

    return valid, length

def clean_oeis(n_seqs = -1, path="/private/home/sdascoli/recur/", dataset_type='very_easy', exclude=[]):
    if "easy" in dataset_type:
        patterns=["easy to produce", "FORMULA", "G.f.", "a(n)"]
    elif dataset_type=='nice':
        patterns=["exceptionally nice"]
    else:
        raise NotImplementedError

    
    n_kept, n_rejected = 0, 0
    lines = []
    with open(path+"OEIS.txt", 'r') as f:
        with open(path+f"OEIS_{dataset_type}.txt", 'w') as w: 
            for i, line in enumerate(f.readlines()):
                if i%100==0: 
                    print(n_kept, n_rejected, end='\t', flush=True)
                    w.flush()
                if n_kept==n_seqs: break
                identifier = line[:8]
                valid, length = check_patterns(identifier, patterns)
                if len(line.split(','))<=41 or not valid: 
                    n_rejected += 1
                    continue
                w.write(line)
                lines.append((length,line))
                n_kept += 1
    f.close(); w.close()
    lines = sorted(lines, key = lambda x : x[0], reverse=True)
    with open(path+f"OEIS_{dataset_type}_ranked.txt", 'w') as w: 
        for line in lines:
            w.write(line[1])    
    return n_kept, n_rejected

def load_oeis(length = 40, path = "/private/home/sdascoli/recur/OEIS_clean.txt"):
    lines = []
    ids   = []
    lens  = []
    with open(path, 'r') as f:
        for line in f.readlines():
            #print(line)
            x = [int(x) for x in line.split(',')[1:-1]]
            if len(x)<length+1: continue
            x = x[:length]
            lens.append(len(x))
            lines.append(x)    
            ids.append(line.split(',')[0])
    print(len(lines), np.mean(lens))
    return lines, ids

if __name__=='__main__':
    clean_oeis(10000)
