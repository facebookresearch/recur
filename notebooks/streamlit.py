import streamlit as st
from notebook_utils import *
import matplotlib.pyplot as plt
import numpy

st.title('Deep symbolic regression for recurrent sequences')
st.markdown("This webpage is a demonstration of the paper *Deep Symbolic Regression for Recurrent Sequences* by Stéphane d'Ascoli, Pierre-Alexandre Kamienny, Guillaume Lample and François Charton.")

INT_PATH = '/checkpoint/fcharton/recur/final/large/float_sequences_False_output_numeric_False_batch_size_32_optimizer_adam_inverse_sqrt,lr=0.0001'
INT_PATH = '/checkpoint/sdascoli/recur/paper/all/float_sequences_False_output_numeric_False_batch_size_32_use_sympy_False'

FLOAT_PATH = '/checkpoint/fcharton/recur/final/large/float_sequences_True_output_numeric_False_batch_size_32_optimizer_adam_inverse_sqrt,lr=0.0001'
FLOAT_PATH = '/checkpoint/sdascoli/recur/paper/all/float_sequences_True_output_numeric_False_batch_size_32_use_sympy_False'

@st.cache(allow_output_mutation=True)
def load_runs():
    new_args = {'cpu':True, 'use_sympy':True, 'simplify':True}
    int_run = read_run(INT_PATH)
    int_env, int_modules, _, _, = load_run(int_run, epoch=None, new_args=new_args)
    float_run = read_run(FLOAT_PATH)
    float_env, float_modules, _, _, = load_run(float_run, epoch=None, new_args=new_args)

    return int_env, int_modules, float_env, float_modules

def get_infix(tree, env):
    infix = env.simplifier.infix_to_sympy(env.simplifier.prefix_to_infix(tree.prefix().replace('idiv','div').split(',')))
    infix = str(latex(infix))
    for i in range(env.params.max_degree+1):
        infix = infix.replace('x_{0 %d}'%i, 'u_{n-%d}'%i)
    infix = 'u_n = '+infix
    return infix

def predict_seq(seq, print_next_term=False):
    
    float_seq = False
    for x in seq:
        if isinstance(x, float):
            float_seq = True
    if float_seq:
        env, modules = float_env, float_modules
    else:
        env, modules = int_env, int_modules

    length = len(seq)

    beam_size = st.select_slider("Beam size", options=[1,3,5,10], key=int(print_next_term))
    n_predictions = st.select_slider("Number of terms to predict", options=[1,5,10,20], key=int(print_next_term)) if print_next_term else length
    # random_seq = st.checkbox("Random", value=False)
    # if random_seq: seq=None
    tree, pred_trees, seq, pred_seqs, scores, mses = predict(env, modules, seq=seq, beam_size=beam_size, verbose=False, pred_len=n_predictions, sort_mse=True)

    pred_tree, pred_seq = pred_trees[0], pred_seqs[0]
    if print_next_term:
        st.write("Predicted next terms:")
        if float_seq:
            st.latex(',\quad '.join([latex_float(x, precision=4) for x in pred_seq[length:length+n_predictions]]))
        else:
            st.latex(', '.join(['{}'.format(x) for x in pred_seq[length:length+n_predictions]]))        
        st.write("Predicted formulas:")

    for pred_seq, pred_tree, mse, score in zip(pred_seqs, pred_trees, mses, scores):
        asymptotic_error = (seq[-1] - pred_seq[-1])/(seq[-1]+1e-10)
        print("{}: {:.4f} {:.2e} {:.2e} ".format(readable_infix(pred_tree), 10**score, mse, asymptotic_error))

    fig, ax = plt.subplots(1,1,figsize=(5,5))

    ax.plot(seq, ls="none", marker='o', color='red', label='Input points')
    already_seen = []
    for i in range(beam_size):
        pred_seq = pred_seqs[i]
        pred_tree = pred_trees[i]
        confidence = 10**scores[i]
        if not pred_tree or np.isnan(np.sum(pred_seq)): continue
        infix = get_infix(pred_tree, env)
        if infix in already_seen: continue
        else:
            already_seen.append(infix)
            infix += '\quad\quad \mathrm{{confidence:}}\ {:.2f}'.format(confidence)
            st.latex(infix)
        if i==0:
            ax.plot(pred_seq, lw=2, color='k', label='Best prediction')
        elif i==1:
            ax.plot(pred_seq, lw=2, alpha=0.2, color='grey', label='Other predictions')
        else:
            ax.plot(pred_seq, lw=2, alpha=0.2, color='grey')
        pred_tree = pred_trees[0].infix()
    ax.legend()
    ax.set_xlabel('$n$')
    ax.set_ylabel('$u_n$')
    if abs(pred_seq[-1])>1e6:
        ax.set_yscale('symlog')

    st.pyplot(fig)

def approx_const(const):

    precision = len(const.split('.')[1])
    const = eval(const)
    # run = read_run(FLOAT_PATH)
    # env, modules, trainer, evaluator = load_run(run, epoch=None)

    seq = [const*n for n in range(1,25)]

    env, modules = float_env, float_modules
    
    beam_size = st.select_slider("Beam size", options=[1,10,50,100], value=100) 
    n_display = st.select_slider("Number to display", options=[1,3,5], value=1) 
    tree, pred_trees, _, pred_seqs, scores, mses = predict(env, modules, seq=seq, beam_size=beam_size, verbose=False, pred_len=0, sort_mse=True)
    st.write("Predicted approximation :")
    already_seen = []
    for i in range(beam_size):
        if len(already_seen)==n_display: break
        pred_tree, pred_seq = pred_trees[i], pred_seqs[i]
        infix = env.simplifier.infix_to_sympy(env.simplifier.prefix_to_infix(pred_tree.prefix().replace(',n',',1').split(',')))
        if infix in already_seen:
            continue
        else:
            already_seen.append(infix)
        latex_infix = str(latex(infix))
        approx = pred_seq[-1]/len(pred_seq)
        error = abs((approx - const)/const)
        error = latex_float(error, precision=1)
        latex_infix += f'= %.{precision+1}f...'%approx
        latex_infix += "\quad\quad \mathrm{{relative\ error}} : {}".format(error)
        st.latex(latex_infix)

int_env, int_modules, float_env, float_modules = load_runs()

st.subheader('Predicting a recurrence formula')
seq = st.text_input('Input: a sequence of numbers separated by commas (can be integers or floats)','1,1,2,3,5,8,13')
seq = [eval(x) for x in seq.rstrip(',').replace(' ','').split(',')]
if abs(seq[0])>10: st.warning('First term is larger in absolute value than 10. This may yield poor results.')
if len(seq)<5: st.warning('Less than 5 terms were given. This may yield poor results.')
if len(seq)>30: st.warning('More than 30 terms were given. This may yield poor results.')
fig = predict_seq(seq, print_next_term=True)

st.subheader('Approximating constants')
const = st.text_input('Constant to approximate','1.64493')
pred = approx_const(const)

st.subheader('Approximating functions')
f = st.text_input('Function to approximate (in Python lambda format, can use all functions from numpy and scipy.special)','lambda x : scipy.special.dawsn(x)')
f = eval(f)
n_points = st.select_slider('Number of input points', options=[10,20,30], value=20)
seq = [f(n) for n in range(n_points)]
fig = predict_seq(seq)
