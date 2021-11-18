import glob
import os
import string
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
import sys
import copy
from pathlib import Path
from sympy import *
import pickle
from collections import defaultdict, OrderedDict
import math
import scipy.special
import warnings
from sklearn.manifold import TSNE

def import_file(full_path_to_module):
    module_dir, module_file = os.path.split(full_path_to_module)
    module_name, module_ext = os.path.splitext(module_file)
    save_cwd = os.getcwd()
    os.chdir(module_dir)
    module_obj = __import__(module_name)
    module_obj.__file__ = full_path_to_module
    globals()[module_name] = module_obj
    os.chdir(save_cwd)
    return module_obj

path = '/private/home/sdascoli/recur/src'
src = import_file(path)
from src.model import build_modules
from src.envs import ENVS, build_env
from src.trainer import Trainer
from src.evaluator import Evaluator, idx_to_infix
from src.envs.generators import RandomRecurrence

############################ GENERAL ############################


def find(array, value):
    idx= np.argwhere(np.array(array)==value)[0,0]
    return idx

def select_runs(runs, params, constraints):
    selected_runs = []
    for irun, run in enumerate(runs):
        keep = True
        for k,v in constraints.items():
            if type(v)!=list:
                v=[v]
            if (not hasattr(run['args'],k)) or (getattr(run['args'],k) not in v):
                keep = False
                break
        if keep:
            selected_runs.append(run)
    selected_params = copy.deepcopy(params)
    for con in constraints:
        selected_params[con]=[constraints[con]]
    return selected_runs, selected_params

def group_runs(runs, finished_only=True):
    runs_grouped = defaultdict(list)
    for run in runs:
        seedless_args = copy.deepcopy(run['args'])
        del(seedless_args.seed)
        del(seedless_args.name)
        if str(seedless_args) not in runs_grouped.keys(): 
            runs_grouped[str(seedless_args)].append(run) # need at least one run
        else:
            if run['finished'] or not finished_only:
                runs_grouped[str(seedless_args)].append(run)
    runs_grouped = list(runs_grouped.values())
    return runs_grouped

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def ordered_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: list(map(float,labels))[k])] )
    #ax.legend(handles, labels, **kwargs)
    return handles, labels

def legend_no_duplicates(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[i+1:]]
    ax.legend(*zip(*unique), **kwargs)
    

############################ RECURRENCE ############################

def readable_infix(tree):
    infix = tree.infix()
    infix = infix.replace('x_0_','x').replace('x_1_','y').replace('x_2_','z')
    infix = infix.replace('add','+').replace('sub','-').replace('mod','%')
    infix = infix.replace('mul','*').replace('idiv','/').replace('div','/').replace('fabs','abs')
    infix = infix.replace('inv','1/').replace('euler_gamma', 'g').replace('rand','w')
    return infix

def sympy_infix(tree):
    infix = readable_infix(tree)
    for i in range(6):
        exec('x{}'.format(i)+'='+"symbols('x{}'.format(i))")
        exec('y{}'.format(i)+'='+"symbols('y{}'.format(i))")
        exec('z{}'.format(i)+'='+"symbols('z{}'.format(i))")
    n, w, e, g = symbols('n w e g')
    init_printing(use_unicode=True)
    return simplify(eval(infix))


def load_run(run, new_args=None):
    
    if new_args is None: new_args = run['args']
    new_args.multi_gpu = False
    new_args.tasks = 'recurrence'
    env = build_env(new_args)
    modules = build_modules(env, new_args)
    trainer = Trainer(modules, env, new_args)
    evaluator = Evaluator(trainer)
    return env, modules, trainer, evaluator

def eval_run(run, new_args=None):
    
    env, modules, trainer, evaluator = load_run(run, new_args=new_args)
    data_types = ["valid1"]
    evaluator.set_env_copies(data_types)
    scores = evaluator.run_all_evals(data_types)   
    return scores

      
def predict(params, series=None, pred_len=None, beam_size=None, beam_length_penalty=None, verbose=False, gen_kwargs={}):
    
    if beam_length_penalty is None: beam_length_penalty = args.beam_length_penalty
    if beam_size is None: beam_size = args.beam_size
        
    if series is None:
        generator = RandomRecurrence(params)
        rng = np.random.RandomState(0)
        rng.seed()
        while True:
            tree, series, _, _ = generator.generate(rng, **gen_kwargs)
            if series is None: continue
            if np.isnan(np.sum(series)): continue
            break
    else:
        tree = None
    
    x = [env.input_encoder.encode(series)]
    x = [torch.LongTensor([env.input_word2id[w] for w in seq]) for seq in x]
    x, x_len = env.batch_sequences(x)
    x, x_len = x.cuda(), x_len.cuda()
    encoded = encoder("fwd", x=x, lengths=x_len, causal=False)
    gen, _, scores = decoder.generate_beam(
                    encoded.transpose(0, 1),
                    x_len,
                    beam_size=beam_size,
                    length_penalty=beam_length_penalty,
                    early_stopping=args.beam_early_stopping,
                    max_len=args.max_len
    )
    score = scores[0].hyp[0][0]
    if verbose:
        for h, hyp in scores[0].hyp:
            tokens = [env.output_id2word[wid] for wid in hyp.tolist()[1:]]
            pred = env.output_encoder.decode(tokens)
            print(readable_infix(pred))
        
    gen = gen.cpu().numpy()[1:-1,0]
    tokens = [env.output_id2word[wid] for wid in gen]
    #pred_tree = env.output_encoder.decode(prefix)[0]
    pred = env.output_encoder.decode(tokens)
    
    if params.output_numeric:
        series.extend(pred[1:])
    
    pred_series = copy.deepcopy(series)
    if pred_len is None: pred_len = len(series)//args.dimension
    for i in range(pred_len):
        if tree: series.extend(tree.val(series))
        if not params.output_numeric: pred_series.extend(pred.val(pred_series))
    
    return tree, pred, series, pred_series, score


############################  OEIS   ############################

def check_oeis(identifier):
    import urllib.request
    uf = urllib.request.urlopen("https://oeis.org/"+identifier)
    text = str(uf.read())
    length = len(text)
    return ('FORMULA' in text) and ('G.f.:' in text), length
    
def clean_oeis(length=20, n_seqs = -1):
    n_kept, n_rejected = 0, 0
    with open("/private/home/sdascoli/recur/OEIS.txt", 'r') as f:
        with open("/private/home/sdascoli/recur/OEIS_clean.txt", 'w') as w: 
            for i, line in enumerate(f.readlines()):
                if i%100==0: 
                    print(n_kept, n_rejected, end='\t')
                    w.flush()
                if n_kept==n_seqs: return n_kept, n_rejected
                identifier = line[:8]
                if len(line.split(','))<30 or (not check_oeis(identifier)): 
                    n_rejected += 1
                    continue
                w.write(line)
                n_kept += 1
    f.close(); w.close()
    return n_kept, n_rejected

def load_oeis():
    lines = []
    ids   = []
    lens  = []
    with open("/private/home/sdascoli/recur/OEIS_clean.txt", 'r') as f:
        for line in f.readlines():
            #print(line)
            x = [int(x) for x in line.split(',')[1:-1]]
            if len(x)<21: continue
            x = x[:30]
            lens.append(len(x))
            lines.append(x)    
            ids.append(line.split(',')[0])
    print(len(lines), np.mean(lens))
    return lines, ids


############################ OPERATOR FAMILIES ############################

### In-domain ###
from src.envs.generators import operators_real
all_ops = list(operators_real.keys())

id_groups = {
             'base': ['add','sub','mul'],
             'div' : ['div'],
             'abs' : ['abs'],
             'sqrt': ['sqrt'],
             'exp' : ['log', 'exp'],
             'trig': ['sin', 'cos', 'tan','arcsin', 'arccos', 'arctan']
            }

### Out-of-domain ###
poly = [k for k in scipy.special.__dir__() if k.startswith('eval')]
od_groups = {
             'poly'   : [func+str(i) for func in poly for i in range(6)],
             'hyper'  : ['sinh','cosh','tanh','arcsinh','arccosh','arctanh'],
             'fresnel': ['erf', 'erfinv', 'wofz', 'dawsn', 'fresnel'],
             'bessel' : ['j0','j1','y0','y1','i0','i0e','i1','i1e','k0','k0e','k1','k1e']
            }