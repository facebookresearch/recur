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
from IPython.display import display
from importlib import reload  # Python 3.4+
import importlib.util

def module_from_file(module_name, file_path):
    print(file_path, module_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def import_file(full_path_to_module):
    module_dir, module_file = os.path.split(full_path_to_module)
    module_name, module_ext = os.path.splitext(module_file)
    save_cwd = os.getcwd()
    os.chdir(module_dir)
    module_obj = __import__(module_name)
    module_obj.__file__ = full_path_to_module
    globals()[module_name] = module_obj
    os.chdir(save_cwd)
    print(module_name, module_obj.__file__, 'hi')
    return module_obj

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
    
    #try: del src
    #except: pass
    #path = '/private/home/sdascoli/recur/src'
        
    path = run['args'].dump_path+'/src'
    src = import_file(path)

    print(src)
    from src.model import build_modules
    from src.envs import ENVS, build_env
    from src.trainer import Trainer
    from src.evaluator import Evaluator, idx_to_infix
    from src.envs.generators import RandomRecurrence
    
    if new_args is None: new_args = copy.deepcopy(run['args'])
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

      
def predict(args, env, modules, series=None, pred_len=None, beam_size=None, beam_length_penalty=None, verbose=False, rec_only=False, nonrec_only=False, gen_kwargs={}):
    
    encoder, decoder = modules["encoder"], modules["decoder"]
    
    if beam_length_penalty is None: beam_length_penalty = args.beam_length_penalty
    if beam_size is None: beam_size = args.beam_size
        
    if series is None:
        generator = env.generator
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
            if nonrec_only:
                if any([token.startswith('x_') for token in tokens]): continue
            if rec_only:
                if not any([token.startswith('x_') for token in tokens]): continue
            pred = env.output_encoder.decode(tokens)
            try:display(env.simplifier.get_simple_infix(pred))
            except:print(pred)
        
    gen = gen.cpu().numpy()[1:-1,0]
    tokens = [env.output_id2word[wid] for wid in gen]
    #pred_tree = env.output_encoder.decode(prefix)[0]
    pred = env.output_encoder.decode(tokens)
    
    if args.output_numeric:
        series.extend(pred[1:])
            
    pred_series = copy.deepcopy(series)
    if pred_len is None: pred_len = len(series)//args.dimension
    for i in range(pred_len):
        if tree: series.extend(tree.val(series))
        if not args.output_numeric: pred_series.extend(pred.val(pred_series))
    
    return tree, pred, series, pred_series, score

def predict_batch(args, env, modules, batch, pred_len=3):
    
    encoder, decoder = modules["encoder"], modules["decoder"]
        
    x = [env.input_encoder.encode(seq) for seq in batch]
    x = [torch.LongTensor([env.input_word2id[w] for w in seq]) for seq in x]
    x, x_len = env.batch_sequences(x)
    x, x_len = x.cuda(), x_len.cuda()
    encoded = encoder("fwd", x=x, lengths=x_len, causal=False)
    gen, _, scores = decoder.generate_beam(
                    encoded.transpose(0, 1),
                    x_len,
                    beam_size=args.beam_size,
                    length_penalty=args.beam_length_penalty,
                    early_stopping=args.beam_early_stopping,
                    max_len=args.max_len
    )

    gens = gen.cpu().numpy()[1:-1,:].T
    tokens = [[env.output_id2word[wid] for wid in gen] for gen in gens]
    tokens = [[token for token in seq if token not in ['PAD', 'EOS']] for seq in tokens]
    preds = [env.output_encoder.decode(token) for token in tokens]

    pred_series = []
    for i, seq in enumerate(batch):
        if not preds[i]: pred_series.append(None)
        else:
            if args.output_numeric: pred_series.append(preds[i])
            else:
                pred_seq = copy.deepcopy(seq)
                for _ in range(pred_len):
                    if pred_seq[-1]>args.max_number: break
                    pred_seq.extend(preds[i].val(pred_seq))
                pred_series.append(pred_seq[len(seq):])
    
    return preds, pred_series



############################ OPERATOR FAMILIES ############################

### In-domain ###
operators_real = {
    'add': 2,
    'sub': 2,
    'mul': 2,
    'div': 2,
    'abs'    :1,
    'inv'    :1,
    'sqr'    :1,
    'sqrt'   :1,
    'log'    :1,
    'exp'    :1,
    'sin'    :1,
    'arcsin' :1,
    'cos'    :1,
    'arccos' :1,
    'tan'    :1,
    'arctan' :1,
}
all_ops = list(operators_real.keys())

id_groups = {
             'base': ['add','sub','mul'],
             'div' : ['div'],
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

############################ ATTENTION MAPS ############################

def plot_attention(args, env, modules):
    
    encoder, decoder = modules["encoder"], modules["decoder"]
    encoder.STORE_OUTPUTS = True
    num_heads = model.n_heads
    num_layers = model.n_layers
    
    new_args = copy.deepcopy(args)
    new_args.series_length = 15
    while True:
        #try:
        tree, pred_tree, series, preds, score = predict(new_args, env, modules, kwargs={'nb_ops':3, 'deg':3, 'length':10})
        break
        #except Exception as e:
            #print(e, end=' ')
    pred, true = readable_infix(pred_tree), readable_infix(tree)
    separations = [idx for idx, val in enumerate(np.array(env.input_encoder.encode(series[:len(series)//2+1]))) if val in ['+','-']]
            
    plt.figure(figsize=(4,4))
    plt.plot(series)
    plt.plot(preds, ls='--')
    plt.title(f'True: {true}\nPred: {pred}\nConfidence: {confidence:.2}', fontsize=10)
    plt.tight_layout()
    plt.savefig(savedir+'attention_plot_{}.pdf'.format(args.real_series))
        
    fig, axarr = plt.subplots(num_layers, num_heads, figsize=(2*num_heads,2*num_layers), constrained_layout=True)        
        
    for l in range(num_layers):
        module = model.attentions[l]
        scores = module.outputs.squeeze()
        
        for head in range(num_heads):                  
            axarr[l][head].matshow(scores[head])
            
            axarr[l][head].set_xticks([]) 
            axarr[l][head].set_yticks([]) 
            #for val in separations: 
            #    axarr[l][head].axvline(val, color='red', lw=.5)
            #    axarr[l][head].axhline(val, color='red', lw=.5)
                
    cols = [r'Head {}'.format(col+1) for col in range(num_heads)]
    rows = ['Layer {}'.format(row+1) for row in range(num_layers)]
    for icol, col in enumerate(cols):
        axarr[0][icol].set_title(col, fontsize=18, pad=10)
    for irow, row in enumerate(rows):
        axarr[irow][0].set_ylabel(row, fontsize=18, labelpad=10)

    plt.tight_layout()
    plt.savefig(savedir+'attention_{}.pdf'.format(args.real_series))
    plt.show()
    
    return tree, pred_tree, series, preds, score
