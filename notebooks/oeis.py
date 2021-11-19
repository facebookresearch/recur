import numpy as np

############################  OEIS   ############################

def check_oeis(identifier):
    import urllib.request
    uf = urllib.request.urlopen("https://oeis.org/"+identifier)
    text = str(uf.read())
    length = len(text)
    return ('FORMULA' in text) and ('G.f.:' in text), length
    
def clean_oeis(n_seqs = -1):
    n_kept, n_rejected = 0, 0
    with open("/private/home/sdascoli/recur/OEIS.txt", 'r') as f:
        with open("/private/home/sdascoli/recur/OEIS_clean.txt", 'w') as w: 
            for i, line in enumerate(f.readlines()):
                if i%100==0: 
                    print(n_kept, n_rejected, end='\t')
                    w.flush()
                if n_kept==n_seqs: return n_kept, n_rejected
                identifier = line[:8]
                valid, length =  check_oeis(identifier)
                if len(line.split(','))<30 or not valid: 
                    n_rejected += 1
                    continue
                w.write(line)
                n_kept += 1
    f.close(); w.close()
    return n_kept, n_rejected

def rank_oeis(n_seqs = -1):
    n_kept, n_rejected = 0, 0
    lines = []
    with open("/private/home/sdascoli/recur/OEIS.txt", 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i%100==0: 
                print(n_kept, n_rejected, end='\t', flush=True)
            if n_kept==n_seqs: break
            identifier = line[:8]
            valid, length =  check_oeis(identifier)
            if len(line.split(','))<30 or not valid: 
                n_rejected += 1
                continue
            lines.append((length, line))
            n_kept += 1
    lines = sorted(lines, key = lambda x : x[0], reverse=True)
    with open("/private/home/sdascoli/recur/OEIS_ranked.txt", 'w') as w: 
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
    rank_oeis(200)
