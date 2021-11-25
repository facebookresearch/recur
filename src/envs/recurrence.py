# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from distutils.log import INFO
from logging import getLogger
import os
import io
import sys
import copy
import json

# import math
import numpy as np
import src.envs.encoders as encoders
import src.envs.generators as generators
import src.envs.simplifiers as simplifiers
from src.envs.generators import all_operators

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import collections
from ..utils import bool_flag

SPECIAL_WORDS = ["EOS", "PAD", "(", ")", "SPECIAL", "OOD_unary_op", "OOD_binary_op", "OOD_constant"]
logger = getLogger()


class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


class RecurrenceEnvironment(object):

    TRAINING_TASKS = {"recurrence"}

    def __init__(self, params):
        self.params = params
        self.float_tolerance = params.float_tolerance
        self.additional_tolerance = [
            float(x) for x in params.more_tolerance.split(",") if len(x) > 0
        ]
        self.generator = generators.RandomRecurrence(params)
        
        if self.params.float_sequences:
            self.input_encoder = encoders.FloatSequences(params)
        else:
            self.input_encoder = encoders.IntegerSequences(params)
        self.input_words = SPECIAL_WORDS+sorted(list(set(self.input_encoder.symbols)))
        self.output_numeric=self.params.output_numeric
        if self.params.output_numeric:
            self.output_encoder = encoders.FloatSequences(params) if self.params.float_sequences else encoders.IntegerSequences(params)
            self.output_words = sorted(list(set(self.output_encoder.symbols)))
        else:
            self.output_encoder = encoders.Equation(params, self.generator.symbols)
            self.output_words = sorted(list(set(self.generator.symbols)))
        self.output_words = self.output_words + SPECIAL_WORDS
        
        if params.use_sympy and not params.output_numeric: self.simplifier = simplifiers.Simplifier(self.output_encoder, self.generator)
        else: self.simplifier = None
    
        # number of words / indices
        self.input_id2word = {i: s for i, s in enumerate(self.input_words)}
        self.output_id2word = {i: s for i, s in enumerate(self.output_words)}
        self.input_word2id = {s: i for i, s in self.input_id2word.items()}
        self.output_word2id = {s: i for i, s in self.output_id2word.items()}

        for ood_unary_op in self.generator.extra_unary_operators:
            self.output_word2id[ood_unary_op]=self.output_word2id["OOD_unary_op"]
        for ood_binary_op in self.generator.extra_binary_operators:
            self.output_word2id[ood_binary_op]=self.output_word2id["OOD_binary_op"]
        if self.generator.extra_constants is not None:
            for c in self.generator.extra_constants:
                self.output_word2id[c]=self.output_word2id["OOD_constant"]
        
        if self.params.float_constants:
            assert self.params.float_sequences, "Constants cannot be float when we consider integer series"

        assert len(self.input_words) == len(set(self.input_words))
        assert len(self.output_words) == len(set(self.output_words))
        self.n_words = params.n_words = len(self.output_words)
        self.eos_index = params.eos_index = self.output_word2id["EOS"]
        self.pad_index = params.pad_index = self.output_word2id["PAD"]

        logger.info(f"vocabulary: {len(self.input_word2id)} input words, {len(self.output_word2id)} output_words")
        # logger.info(f"output words: {self.output_word2id.keys()}")
        
    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(
            self.pad_index
        )
        #assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1 : lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def input_to_infix(self, lst, str_array=True):
        m = self.input_encoder.decode(lst)
        if m is None:
            return "Invalid"
        if str_array:
            return np.array2string(np.array(m))
        else:
            return np.array(m)


    def output_to_infix(self, lst, str_array=True):
        m = self.output_encoder.decode(lst)
        if m is None:
            return "Invalid"
        if self.params.output_numeric:
            if str_array:
                return np.array2string(np.array(m))
            else:
                return np.array(m)
        else:
            return m.infix()

    def gen_expr(self, train, input_length_modulo=-1, nb_ops=None):
        
        length=self.params.max_len if input_length_modulo!=-1 and not train else None
        tree, series, predictions, n_input_points = self.generator.generate(rng=self.rng, nb_ops=nb_ops, prediction_points=True,length=length)
        if tree is None:
            return None, None, None, None
        n_ops = tree.get_n_ops()
        n_recurrence_degree = max(tree.get_recurrence_degrees())
        if self.simplifier is not None: tree = self.simplifier.simplify_tree(tree)
        
        ##compute predictions even in symbolic case
        
        if not train:
            ##TODO: add noise to predictions
            if self.params.eval_noise:
                noise = self.params.eval_noise
                if self.params.float_sequences:
                    noise = self.params.eval_noise * np.random.randn(len(series))
                else:
                    noise = np.random.randint(size=(len(series),), low=-self.params.eval_noise, high=self.params.eval_noise+1).astype(int)
                if self.params.eval_noise_type=="additive":
                    series = (np.array(series)+noise).tolist()
                elif self.params.eval_noise_type=="multiplicative":
                    series = (np.array(series)*(1+noise)).tolist()
                else:
                    raise NotImplementedError

        ending = np.array(series[-5:])
        gaps = abs(ending[1:]-ending[:-1])
        if len(set(gaps))<2: return None, None, None, None # discard uninteresting series            

        x = self.input_encoder.encode(series)
        if self.params.output_numeric:
            y = self.output_encoder.encode(predictions)
        else:
            y = self.output_encoder.encode(tree)

        max_token_len=self.params.max_token_len
        if max_token_len > 0 and (len(x) >= max_token_len or len(y) >= max_token_len):
            return None, None, None, None
        
        if input_length_modulo!=-1 and not train:
            indexes_to_remove = [i*input_length_modulo for i in range((self.params.max_len-self.params.min_len)//input_length_modulo+1)]
        else:
            indexes_to_remove = [0]

        x, y = [], []
        info = {"n_input_points":[], "n_ops": [], "n_recurrence_degree": []}
        for idx in indexes_to_remove:
            #if self.params.output_numeric:
            #    if idx==0:
            #        input_seq = series+predictions[:1]
            #    elif idx==1:
            #        input_seq=series
            #    else:
            #        input_seq=series[:-idx+1]
            #else:
            input_seq=series[:-idx] if idx>0 else series
            _x = self.input_encoder.encode(input_seq)
            if self.params.output_numeric:
                if idx>self.params.n_predictions:
                    _predictions=series[-idx:-idx+self.params.n_predictions]
                elif idx==self.params.n_predictions:
                    _predictions=series[-idx:]
                elif idx==0:
                    _predictions=predictions
                else:
                    _predictions=series[-idx:]+predictions[:self.params.n_predictions-idx]
                _y = self.output_encoder.encode(_predictions)
            else:
                _y = self.output_encoder.encode(tree)
            x.append(_x)
            y.append(_y)

            info["n_input_points"].append(n_input_points-idx)
            info["n_ops"].append(n_ops)
            info["n_recurrence_degree"].append(n_recurrence_degree)
            
        return x, y, tree, info

    def code_class(self, tree):
        return {"ops": tree.get_n_ops()}
        
    def decode_class(self, nb_ops):
        return nb_ops

    def check_prediction(self, src, tgt, hyp, tree, n_predictions=5):
        src = self.input_encoder.decode(src)
        eq_hyp = self.output_encoder.decode(hyp)
        if self.params.output_numeric:
            eq_tgt = self.output_encoder.decode(tgt)
            if eq_hyp is None or np.nan in eq_hyp or len(eq_tgt)!=len(eq_hyp):
                return [-1 for _ in range(n_predictions)]
        else:
            eq_tgt = tree
            if eq_hyp is None:
                return [-1 for _ in range(n_predictions)]
        if self.params.output_numeric:
            error = self.generator.evaluate_numeric(tgt=eq_tgt, hyp=eq_hyp)                    
        else:
            if eq_tgt is None: # When we don't have the ground truth, test on last terms
                error = self.generator.evaluate_without_target(src, eq_hyp, n_predictions)
            else:
                error = self.generator.evaluate(src, eq_tgt, eq_hyp, n_predictions)
        return error

    def create_train_iterator(self, task, data_path, params, args={}):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")
        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
            **args
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 5000),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size, input_length_modulo, **args
    ):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=(
                None
                if data_path is None
                else data_path[task][int(data_type[5:])]
            ),
            size=size,
            type=data_type,
            input_length_modulo=input_length_modulo,
            **args
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        
        parser.add_argument("--output_numeric", type=bool_flag, default=False,
                            help="Whether we learn to predict numeric values or a symbolic expression")
        parser.add_argument("--float_sequences", type=bool_flag, default=False,
                            help="Whether to use float sequences rather than integer sequences")
        parser.add_argument("--use_sympy", type=bool_flag, default=True,
                            help="Whether to use sympy parsing (basic simplification)")
        parser.add_argument("--simplify", type=bool_flag, default=False,
                            help="Whether to use further sympy simplification")
        
        # encoding
        parser.add_argument("--operators_to_remove", type=str, default="",
                            help="Which operator to remove")
        parser.add_argument("--required_operators", type=str, default="",
                            help="Which operator to remove")    
        parser.add_argument("--extra_unary_operators", type=str, default="",
                            help="Extra unary operator to add to data generation")    
        parser.add_argument("--extra_binary_operators", type=str, default="",
                            help="Extra binary operator to add to data generation")    
        parser.add_argument("--float_constants", type=float, default=None,
                            help="Use float constants instead of ints") 
        parser.add_argument("--extra_constants", type=str, default=None,
                            help="Additional int constants floats instead of ints") 


        parser.add_argument("--dimension", type=int, default=1,
                            help="Number of variables")
        parser.add_argument("--float_precision", type=int, default=3,
                            help="Number of digits in the mantissa")
        parser.add_argument("--max_exponent", type=int, default=100,
                            help="Maximal order of magnitude")
        parser.add_argument("--int_base", type=int, default=10000,
                            help="Integer base used when encoding sequences")
        parser.add_argument("--max_number", type=int, default=1e100,
                            help="Maximal order of magnitude")
        parser.add_argument("--max_token_len", type=int, default=0, help="max size of tokenized sentences, 0 is no filtering")

        #generator 
        parser.add_argument("--max_int", type=int, default=10,
                            help="Maximal integer in symbolic expressions")
        parser.add_argument("--max_degree", type=int, default=6,
                            help="Number of elements in the sequence the next term depends on")
        parser.add_argument("--max_ops", type=int, default=10,
                            help="Number of unary or binary operators")
        parser.add_argument("--min_op_prob", type=float, default=0.01,
                            help="Minimum probability of generating an example with given n_op, for our curriculum strategy")
        parser.add_argument("--max_len", type=int, default=30,
                            help="Max number of terms in the series")
        parser.add_argument("--min_len", type=int, default=5,
                            help="Min number of terms in the series")
        parser.add_argument("--init_scale", type=int, default=10,
                            help="Scale of the initial terms of the series")
        parser.add_argument("--prob_const", type=float, default=1/3,
                            help="Probability to generate integer in leafs")
        parser.add_argument("--prob_n", type=float, default=1/3,
                            help="Probability to generate n in leafs")
        parser.add_argument("--prob_rand", type=float, default=0.,
                            help="Probability to generate n in leafs")
       
        # evaluation
        parser.add_argument("--float_tolerance", type=float, default=1e-10,
                            help="error tolerance for float results")
        parser.add_argument("--more_tolerance", type=str, default="0.0,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1", 
                            help="additional tolerance limits")
        parser.add_argument("--n_predictions", type=int, default=10, 
                            help="number of next terms to predict")



class EnvDataset(Dataset):
    def __init__(self, env, task, train, params, path, size=None, type=None,input_length_modulo=-1, **args):
        super(EnvDataset).__init__()
        self.env = env
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        self.type = type
        self.input_length_modulo=input_length_modulo

        if "nb_ops_prob" in args:
            self.nb_ops_prob = args["nb_ops_prob"]
        else:
            self.nb_ops_prob = None
        if "test_env_seed" in args:
            self.test_env_seed = args["test_env_seed"]
        else:
            self.test_env_seed = None 
        if "env_info" in args:
            self.env_info=args["env_info"]
        else:
            self.env_info=None

        assert task in RecurrenceEnvironment.TRAINING_TASKS
        assert size is None or not self.train
        assert not params.batch_load or params.reload_size > 0
        self.remaining_data=0
        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        self.batch_load = params.batch_load
        self.reload_size = params.reload_size
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0

        # generation, or reloading from file
        if path is not None:
            assert os.path.isfile(path), "{} not found".format(path)
            if params.batch_load and self.train:
                self.load_chunk()
            else:
                logger.info(f"Loading data from {path} ...")
                with io.open(path, mode="r", encoding="utf-8") as f:
                    # either reload the entire file, or the first N lines
                    # (for the training set)
                    if not train:
                        lines = []
                        for i, line in enumerate(f):
                            lines.append(json.loads(line.rstrip()))
                    else:
                        lines = []
                        for i, line in enumerate(f):
                            if i == params.reload_size:
                                break
                            if i % params.n_gpu_per_node == params.local_rank:
                                #lines.append(line.rstrip())
                                lines.append(json.loads(line.rstrip()))
                #self.data = [xy.split("=") for xy in lines]
                #self.data = [xy for xy in self.data if len(xy) == 3]
                self.data=lines
                logger.info(f"Loaded {len(self.data)} equations from the disk.")

        # dataset size: infinite iterator for train, finite for valid / test
        # (default of 10000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            self.size = 10000 if path is None else len(self.data)
        else:
            assert size > 0
            self.size = size

    def load_chunk(self):
        self.basepos = self.nextpos
        logger.info(
            f"Loading data from {self.path} ... seekpos {self.seekpos}, "
            f"basepos {self.basepos}"
        )
        endfile = False
        with io.open(self.path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines = []
            for i in range(self.reload_size):
                line = f.readline()
                if not line:
                    endfile = True
                    break
                if i % self.n_gpu_per_node == self.local_rank:
                    lines.append(line.rstrip().split("|"))
            self.seekpos = 0 if endfile else f.tell()

        self.data = [xy.split("\t") for _, xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]
        self.nextpos = self.basepos + len(self.data)
        logger.info(
            f"Loaded {len(self.data)} equations from the disk. seekpos {self.seekpos}, "
            f"nextpos {self.nextpos}"
        )
        if len(self.data) == 0:
            self.load_chunk()

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y, tree, infos = zip(*elements)
        info_tensor = {info_type: torch.LongTensor([info[info_type] for info in infos]) for info_type in infos[0].keys()} #[self.env.code_class(treei) for _,_,treei in zip(x, y, tree)]
        x = [torch.LongTensor([self.env.input_word2id[w] for w in seq]) for seq in x]
        y = [torch.LongTensor([self.env.output_word2id[w] for w in seq]) for seq in y]
        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)
        return (x, x_len), (y, y_len), tree, info_tensor

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self.env, "rng"):
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = [worker_id, self.global_rank, self.env_base_seed]
            if self.env_info is not None:
                seed+=[self.env_info]
            self.env.rng = np.random.RandomState(seed)
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{seed} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = self.test_env_seed  if "valid" in self.type else 0
            self.env.rng = np.random.RandomState(seed)
            logger.info(
                "Initialized {} generator, with seed {} (random state: {})".format(self.type, seed, self.env.rng)    
            )
            #print(self.generate_sample())

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """s
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        idx = index
        if self.train:
            if self.batch_load:
                if index >= self.nextpos:
                    self.load_chunk()
                idx = index - self.basepos
            else:
                index = self.env.rng.randint(len(self.data))
                idx = index
        x = self.data[idx]
        x1 = x["x1"].split(" ")
        x2 = x["x2"].split(" ")
        tree_str = x["tree"]
        tree = tree_str  ##TODO : if we want the representation not infix
        infos = {}
        for col in x:
            if col not in ["x1", "x2", "tree"]:
                infos[col]=int(x[col])
        return x1, x2, tree, infos 
        
        #x, y = self.data[idx]
        #x = x.split()
        #y = y.split()
        #assert len(x) >= 1 and len(y) >= 1
        #return x, y

    def generate_sample(self):
        """
        Generate a sample.
        """

        def select_index(dico, idx):
            new_dico={}
            for k in dico.keys():
                new_dico[k]=dico[k][idx]
            return new_dico

        if self.remaining_data == 0:
            while True:
                try:
                    if self.task == "recurrence":
                        self._x,self._y,self.tree, self.infos = self.env.gen_expr(self.train, input_length_modulo=self.input_length_modulo, nb_ops=self.nb_ops_prob)
                    else:
                        raise Exception(f"Unknown data type: {self.task}")
                    if self._x is None or self._y is None:
                        continue # discard problematic series
                    break
                except Exception as e:
                    if False: logger.error(
                        'An unknown exception of type {0} occurred for worker {4} in line {1} for expression "{2}". Arguments:{3!r}.'.format(
                            type(e).__name__,
                            sys.exc_info()[-1].tb_lineno,
                            "F",
                            e.args,
                            self.get_worker_id(),
                        )
                    )
                    continue
            self.remaining_data=len(self._x)

        x,y,tree,info=self._x[-self.remaining_data], self._y[-self.remaining_data], self.tree, select_index(self.infos,-self.remaining_data)
        self.remaining_data-=1
        self.count += 1
        return x, y, tree, info
