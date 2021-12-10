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
        
def permute(array, indices):
    return [array[idx] for idx in indices]

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

def read_run(path):
    
    run = {}
    args = pickle.load(open(path+'/params.pkl', 'rb'))
    run['args'] = args
    if 'use_sympy' not in args:
        setattr(args,'use_sympy',False)
    if 'mantissa_len' not in args:
        setattr(args,'mantissa_len',1)
    if 'train_noise' not in args:
        setattr(args,'train_noise', 0)
    setattr(args, 'extra_constants', '')
    run['logs'] = []
    run['num_params'] = []
    logfile = path+'/train.log'
    f = open(logfile, "r")
    for line in f.readlines():
        if '__log__' in line:
            log = eval(line[line.find('{'):].rstrip('\n'))
            if not run['logs']: run['logs'].append(log)
            else: 
                if log['valid1_recurrence_beam_acc'] != run['logs'][-1]['valid1_recurrence_beam_acc']: run['logs'].append(log)
    f.close()
    args.output_dir = Path(path)
    return run
    
def load_run(run, new_args={}, epoch=None):
    
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
    
    final_args = copy.deepcopy(run['args'])
    for arg, val in new_args.items():
        setattr(final_args,arg,val)
    final_args.multi_gpu = False
    
    env = build_env(final_args)
    modules = build_modules(env, final_args)
        
    trainer = Trainer(modules, env, final_args)
    evaluator = Evaluator(trainer)
    
    if epoch is not None:
        print(f"Reloading epoch {epoch}")
        checkpoint_path = os.path.join(final_args.dump_path,f'periodic-{epoch}.pth')
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        new_checkpoint = {}
        for k, module in modules.items():
            weights = {k.partition('.')[2]:v for k,v in checkpoint[k].items()}
            module.load_state_dict(weights)
            module.eval()
    
    return env, modules, trainer, evaluator

def eval_run(run, new_args=None):
    
    env, modules, trainer, evaluator = load_run(run, new_args=new_args)
    data_types = ["valid1"]
    evaluator.set_env_copies(data_types)
    scores = evaluator.run_all_evals(data_types)   
    return scores

      
def predict(env, modules, seq=None, pred_len=None, beam_size=None, beam_length_penalty=None, verbose=False, rec_only=False, nonrec_only=False, sort_mse=True, gen_kwargs={}):
    
    args = env.params
    encoder, decoder = modules["encoder"], modules["decoder"]
    
    if beam_length_penalty is None: beam_length_penalty = args.beam_length_penalty
    if beam_size is None: beam_size = args.beam_size
        
    if seq is None:
        generator = env.generator
        rng = np.random
        rng.seed()
        while True:
            tree, seq, _, _ = generator.generate(rng, **gen_kwargs)
            if seq is None: continue
            if np.isnan(np.sum(seq)): continue
            break
    else:
        tree = None
    
    x = [env.input_encoder.encode(seq)]
    x = [torch.LongTensor([env.input_word2id[w] for w in s]) for s in x]
    x, x_len = env.batch_sequences(x)
    if not args.cpu:
        x, x_len = x.cuda(), x_len.cuda()
    encoded = encoder("fwd", x=x, lengths=x_len, causal=False)
    gen, _, gens = decoder.generate_beam(
                    encoded.transpose(0, 1),
                    x_len,
                    beam_size=beam_size,
                    length_penalty=beam_length_penalty,
                    early_stopping=args.beam_early_stopping,
                    max_len=args.max_len
    )
    
    pred_trees, pred_seqs, scores, mses = [], [], [], []
            
    for score, hyp in gens[0].hyp:
        scores.append(score)
        tokens = [env.output_id2word[wid] for wid in hyp.tolist()[1:]]
        if nonrec_only:
            if any([token.startswith('x_') for token in tokens]): continue
        if rec_only:
            if not any([token.startswith('x_') for token in tokens]): continue
        pred_tree = env.output_encoder.decode(tokens)
        if pred_tree is None: continue
        degs = pred_tree.get_recurrence_degrees()
        if verbose:
            try:display(env.simplifier.get_simple_infix(pred_tree))
            except:print(pred_tree)

        if args.output_numeric:
            seq.extend(pred[1:])
            
        pred_seq = copy.deepcopy(seq)[:max(degs)*args.dimension]
        while len(pred_seq)<len(seq):
            if not args.output_numeric: pred_seq.extend(pred_tree.val(pred_seq, deterministic=True))
                
        mse = sum([(x-y)**2 for x,y in zip(seq, pred_seq)])
        mses.append(mse)
                    
        if pred_len is None: pred_len = len(seq)//args.dimension
        for i in range(pred_len):
            if tree: seq.extend(tree.val(seq, deterministic=True))
            if not args.output_numeric: pred_seq.extend(pred_tree.val(pred_seq, deterministic=True))
                
        pred_trees.append(pred_tree)
        pred_seqs.append(pred_seq)
        
    if sort_mse:
        order = np.argsort(mses)
        pred_trees, pred_seqs, scores, mses = permute(pred_trees, order), permute(pred_seqs, order), permute(scores, order), permute(mses, order)
    
    return tree, pred_trees, seq, pred_seqs, scores, mses

def predict_batch(env, modules, batch, pred_len=3):
    
    args = env.params
    encoder, decoder = modules["encoder"], modules["decoder"]
        
    x = [env.input_encoder.encode(seq) for seq in batch]
    x = [torch.LongTensor([env.input_word2id[w] for w in seq]) for seq in x]
    x, x_len = env.batch_sequences(x)
    if not args.cpu:
        x, x_len = x.cuda(), x_len.cuda()
    encoded = encoder("fwd", x=x, lengths=x_len, causal=False)
    gen, _, gens = decoder.generate_beam(
                    encoded.transpose(0, 1),
                    x_len,
                    beam_size=args.beam_size,
                    length_penalty=args.beam_length_penalty,
                    early_stopping=args.beam_early_stopping,
                    max_len=args.max_len
    )

    #gens = gen.cpu().numpy()[1:-1,:].T
    #tokens = [[env.output_id2word[wid] for wid in gen] for gen in gens]
    #tokens = [[token for token in seq if token not in ['PAD', 'EOS']] for seq in tokens]
    #pred_trees = [env.output_encoder.decode(token) for token in tokens]
    
    pred_trees = []
    
    if args.output_numeric:
        gens = gen.cpu().numpy()[1:-1,:].T
        tokens = [[env.output_id2word[wid] for wid in gen] for gen in gens]
        tokens = [[token for token in seq if token not in ['PAD', 'EOS']] for seq in tokens]
        pred_trees = [env.output_encoder.decode(token) for token in tokens]
    else:
        for i, gen in enumerate(gens):
            seq = batch[i]
            trees = []
            mses = []
            for score, hyp in gen.hyp:
                tokens = [env.output_id2word[wid] for wid in hyp.tolist()[1:]]
                pred_tree = env.output_encoder.decode(tokens)
                if pred_tree is None: continue
                degs = pred_tree.get_recurrence_degrees()
                pred_seq = copy.deepcopy(seq)[:max(degs)*args.dimension]
                while len(pred_seq)<len(seq):
                    pred_seq.extend(pred_tree.val(pred_seq, deterministic=True))
                mse = sum([(x-y)**2 for x,y in zip(seq, pred_seq)])
                trees.append(pred_tree)
                mses.append(mse)
            if not trees: pred_trees.append(None)
            else: pred_trees.append(trees[np.argmin(mses)])        

    pred_seqs = []
    for i, seq in enumerate(batch):
        if not pred_trees[i]: pred_seqs.append(None)
        else:
            if args.output_numeric: pred_seqs.append(pred_trees[i])
            else:
                pred_seq = copy.deepcopy(seq)
                for _ in range(pred_len):
                    if pred_seq[-1]>args.max_number: break
                    pred_seq.extend(pred_trees[i].val(pred_seq, deterministic=True))
                pred_seqs.append(pred_seq[len(seq):])
    
    return pred_trees, pred_seqs


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
