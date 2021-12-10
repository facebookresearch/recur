import streamlit as st
from notebook_utils import *
import matplotlib.pyplot as plt

st.title('Predicting recurrent sequences')

INT_PATH = '/checkpoint/fcharton/recur/final/large/float_sequences_False_output_numeric_False_batch_size_32_optimizer_adam_inverse_sqrt,lr=0.0001'
INT_PATH = '/checkpoint/sdascoli/recur/paper/large/float_sequences_False_output_numeric_False_batch_size_32_optimizer_adam_inverse_sqrt,lr=0.0001_mantissa_len_3_float_precision_11'
FLOAT_PATH = '/checkpoint/fcharton/recur/final/large/float_sequences_True_output_numeric_False_batch_size_32_optimizer_adam_inverse_sqrt,lr=0.0001'
FLOAT_PATH = '/checkpoint/sdascoli/recur/paper/large/float_sequences_True_output_numeric_False_batch_size_32_mantissa_len_3_float_precision_11_prob_rand_0_use_sympy_False'

@st.cache()
def load_runs():
    new_args = {'cpu':True}
    int_run = read_run(INT_PATH)
    int_env, int_modules, _, _, = load_run(int_run, epoch=None, new_args=new_args)
    float_run = read_run(FLOAT_PATH)
    float_env, float_modules, _, _, = load_run(float_run, epoch=None, new_args=new_args)

    return int_env, int_modules, float_env, float_modules

def predict_seq(seq):
    
    float_seq = False
    seq = [eval(x) for x in seq.split(',')]
    for x in seq:
        if isinstance(x, float):
            float_seq = True
    if float_seq:
        env, modules = float_env, float_modules
    else:
        env, modules = int_env, int_modules

    length = len(seq)
    tree, pred_trees, _, pred_seqs, scores, mses = predict(env, modules, seq=seq, beam_size=5, verbose=False, pred_len=length//2, sort_mse=True)

    pred_tree, pred_seq = pred_trees[0], pred_seqs[0]
    
    infix = str(latex(env.simplifier.get_simple_infix(pred_tree)))
    for i in range(env.params.max_degree):
        infix = infix.replace('x_{0 %d}'%i, 'u_{n-%d}'%i)
        print(infix)
    infix = 'u_n = '+infix

    st.write("Predicted formula: ")
    st.latex(infix)
    st.write("Predicted next term: ", pred_seqs[0][length])
    # st.write("Confidence: ",10**

    for pred_seq, pred_tree, mse, score in zip(pred_seqs, pred_trees, mses, scores):
        asymptotic_error = (seq[-1] - pred_seq[-1])/seq[-1]
        print("{}: {:.4f} {:.2e} {:.2e} ".format(readable_infix(pred_tree), 10**score, mse, asymptotic_error))

    fig, ax = plt.subplots(1,1,figsize=(5,5))

    n_plot = 5
    for i in range(min(n_plot, len(pred_seqs))):
        pred_seq = pred_seqs[i]
        confidence = 10**scores[i]
        # pred = readable_infix(pred)
        if i==0:
            ax.plot(pred_seq, lw=2, color='k')
        else:
            ax.plot(pred_seq, color=plt.cm.viridis(i/n_plot), lw=1, alpha=0.2)
        pred_tree = pred_trees[0].infix()
    ax.plot(seq, ls="none", marker='o', color='red')
    ax.set_xlabel('$n$')
    ax.set_ylabel('$u_n$')

    st.pyplot(fig)

def approx_const(const):

    const = eval(const)
    # run = read_run(FLOAT_PATH)
    # env, modules, trainer, evaluator = load_run(run, epoch=None)

    seq = [const*n for n in range(30)]
    
    tree, pred_trees, _, pred_seqs, scores, mses = predict(float_env, float_modules, seq=seq, beam_size=5, verbose=False, pred_len=10, sort_mse=True)
    pred = pred_trees[0].prefix()
    approx = pred_seqs[0][-1]/len(pred_seqs[0])
    st.write(pred, approx)

    return pred

int_env, int_modules, float_env, float_modules = load_runs()

seq = st.text_input('Input sequence','1,1,2,3,5,8,13')
fig = predict_seq(seq)

const = st.text_input('Constant to approximate','1.644')
pred = approx_const(const)
