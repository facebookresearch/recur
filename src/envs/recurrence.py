# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import io
import sys
import copy

# import math
import numpy as np
import src.envs.encoders as encoders
import src.envs.generators as generators
from src.envs.generators import all_operators

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import collections
from ..utils import bool_flag

SPECIAL_WORDS = ["EOS", "PAD", "(", ")", "SPECIAL"]
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
        
        if self.params.real_series:
            self.input_encoder = encoders.RealSeries(params)
        else:
            self.input_encoder = encoders.IntegerSeries(params)
        self.input_words = SPECIAL_WORDS+sorted(list(set(self.input_encoder.symbols)))

        if self.params.output_numeric:
            self.output_encoder = encoders.RealSeries(params) if self.params.real_series else encoders.IntegerSeries(params)
            self.output_words = sorted(list(set(self.output_encoder.symbols)))
        else:
            self.output_encoder = encoders.Equation(params)
            self.output_words = sorted(list(set(self.generator.symbols)))
        self.output_words = SPECIAL_WORDS+self.output_words
    
        # number of words / indices
        self.input_id2word = {i: s for i, s in enumerate(self.input_words)}
        self.output_id2word = {i: s for i, s in enumerate(self.output_words)}

    
       
        self.input_word2id = {s: i for i, s in self.input_id2word.items()}
        self.output_word2id = {s: i for i, s in self.output_id2word.items()}
        assert len(self.input_words) == len(set(self.input_words))
        assert len(self.output_words) == len(set(self.output_words))
        self.n_words = params.n_words = len(self.output_words)
        self.eos_index = params.eos_index = self.output_word2id["EOS"]
        self.pad_index = params.pad_index = self.output_word2id["PAD"]

        logger.info(f"vocabulary: {len(self.input_word2id)} input words, {len(self.output_word2id)} output_words")
        logger.info(f"output words: {self.output_word2id.keys()}")
        
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
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1 : lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def input_to_infix(self, lst):
        m = self.input_encoder.decode(lst)
        if m is None:
            return "Invalid"
        return np.array2string(np.array(m))

    def output_to_infix(self, lst):
        m = self.output_encoder.decode(lst)
        if m is None:
            return "Invalid"
        if self.params.output_numeric:
            return np.array2string(np.array(m))
        else:
            return m.infix()

    def gen_expr(self, train):
        tree, series, predictions = self.generator.generate(rng=self.rng, prediction_points=self.params.output_numeric)
        if tree is None or np.isnan(series[-1]):# or len(series)<self.params.series_length:
            return None, None, None
        
        if not train:
            ##TODO: add noise to predictions
            if self.params.eval_noise:
                noise = self.params.eval_noise
                if self.params.real_series:
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
        if len(set(gaps))<2: return None, None, None # discard uninteresting series
        
        x = self.input_encoder.encode(series)
        if self.params.output_numeric:
            y = self.output_encoder.encode(predictions)
        else:
            y = self.output_encoder.encode(tree)

        return x, y, tree

    def code_class(self, tree):
        return tree.get_n_ops()
        
    def decode_class(self, nb_ops):
        return nb_ops

    def check_prediction(self, src, tgt, hyp, n_predictions=5):
        src = self.input_encoder.decode(src)
        eq_hyp = self.output_encoder.decode(hyp)
        if self.params.output_numeric:
            if eq_hyp is None or np.nan in eq_hyp:
                return -1
        else:
            if eq_hyp is None:
                return -1
        eq_tgt = self.output_encoder.decode(tgt)
        if self.params.output_numeric:
            error = self.generator.evaluate_numerical(tgt=eq_tgt, hyp=eq_hyp)
        else:
            if eq_tgt is None: # When we don't have the ground truth, test on last terms
                error = self.generator.evaluate_without_target(src, eq_hyp, n_predictions)
            else:
                error = self.generator.evaluate(src, eq_tgt, eq_hyp, n_predictions)
        return error

    def create_train_iterator(self, task, data_path, params):
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
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
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
        self, data_type, task, data_path, batch_size, params, size
    ):
        """
        Create a dataset for this environment.
        """
        assert data_type in ["valid", "test"]
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=(
                None
                if data_path is None
                else data_path[task][1 if data_type == "valid" else 2]
            ),
            size=size,
            type=data_type,
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
        
        parser.add_argument("--output_numeric", type=bool_flag, default=True,
                            help="Whether we learn to predict numeric values or a symbolic expression")
        
        # encoding
        parser.add_argument("--real_series", type=bool_flag, default=True,
                            help="Whether to use real series rather than integer series")
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

        #generator 
        parser.add_argument("--max_int", type=int, default=10,
                            help="Maximal integer in symbolic expressions")
        parser.add_argument("--max_degree", type=int, default=6,
                            help="Number of elements in the sequence the next term depends on")
        parser.add_argument("--max_ops", type=int, default=6,
                            help="Number of unary or binary operators")
        parser.add_argument("--max_len", type=int, default=30,
                            help="Max number of terms in the series")
        parser.add_argument("--init_scale", type=int, default=10,
                            help="Scale of the initial terms of the series")
        parser.add_argument("--prob_const", type=float, default=1/3,
                            help="Probability to generate integer in leafs")
        parser.add_argument("--prob_n", type=float, default=1/3,
                            help="Probability to generate n in leafs")
        parser.add_argument("--prob_rand", type=float, default=0.,
                            help="Probability to generate n in leafs")
       
        # evaluation
        parser.add_argument("--float_tolerance", type=float, default=0.001,
                            help="error tolerance for float results")
        parser.add_argument("--more_tolerance", type=str, default="0.01,0.1", 
                            help="additional tolerance limits")
        parser.add_argument("--n_predictions", type=int, default=5, 
                            help="number of next terms to predict")



class EnvDataset(Dataset):
    def __init__(self, env, task, train, params, path, size=None, type=None):
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
        assert task in RecurrenceEnvironment.TRAINING_TASKS
        assert size is None or not self.train
        assert not params.batch_load or params.reload_size > 0

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
            assert os.path.isfile(path)
            if params.batch_load and self.train:
                self.load_chunk()
            else:
                logger.info(f"Loading data from {path} ...")
                with io.open(path, mode="r", encoding="utf-8") as f:
                    # either reload the entire file, or the first N lines
                    # (for the training set)
                    if not train:
                        lines = [line.rstrip() for line in f]
                    else:
                        lines = []
                        for i, line in enumerate(f):
                            if i == params.reload_size:
                                break
                            if i % params.n_gpu_per_node == params.local_rank:
                                lines.append(line.rstrip())
                self.data = [xy.split("=") for xy in lines]
                self.data = [xy for xy in self.data if len(xy) == 2]
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
        x, y, tree = zip(*elements)
        code = [self.env.code_class(treei) for xi,yi,treei in zip(x, y, tree)]
        x = [torch.LongTensor([self.env.input_word2id[w] for w in seq]) for seq in x]
        y = [torch.LongTensor([self.env.output_word2id[w] for w in seq]) for seq in y]
        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)
        return (x, x_len), (y, y_len), torch.LongTensor(code)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self.env, "rng"):
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.env.rng = np.random.RandomState(
                [worker_id, self.global_rank, self.env_base_seed]
            )
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{[worker_id, self.global_rank, self.env_base_seed]} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            self.env.rng = np.random.RandomState(None if self.type == "valid" else 0)

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
        """
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
        x, y = self.data[idx]
        x = x.split()
        y = y.split()
        assert len(x) >= 1 and len(y) >= 1
        return x, y

    def generate_sample(self):
        """
        Generate a sample.
        """
        while True:
            try:
                if self.task == "recurrence":
                    x,y,tree = self.env.gen_expr(self.train)
                else:
                    raise Exception(f"Unknown data type: {self.task}")
                if x is None or y is None:
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
        self.count += 1

        return x, y, tree
