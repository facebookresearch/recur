from abc import ABC, abstractmethod
import numpy as np
import math

from numpy.compat.py3k import npy_load_module
from wrapt_timeout_decorator import *

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
#    'sinh'   :1,
#    'arcsinh':1,
#    'cosh'   :1,
#    'arccosh':1,
#    'tanh'   :1,
#    'arctanh':1,
}

operators_int = {
    'add': 2,
    'sub': 2,
    'mul': 2,
    'idiv':2,
    'mod': 2,
    'abs': 1,
    'sqr': 1,
}

math_constants = ['e','pi','euler_gamma']

all_operators = {**operators_real, **operators_int}

class Node():
    def __init__(self, value, params, children=None):
        self.value = value
        self.children = children if children else []
        self.params = params

    def push_child(self, child):
        self.children.append(child)

    def prefix(self):
        s = str(self.value)
        for c in self.children:
            s += ',' + c.prefix()
        return s

    # export to latex qtree format: prefix with \Tree, use package qtree
    def qtree_prefix(self):
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self):
        nb_children = len(self.children)
        if nb_children <= 1:
            s = str(self.value)
            if nb_children == 1:
                if s == 'sqr': s = '(' + self.children[0].infix() + ')**2'
                else: s = s + '(' + self.children[0].infix() + ')'
            return s
        s = '(' + self.children[0].infix()
        for c in self.children[1:]:
            s = s + ' ' + str(self.value) + ' ' + c.infix()
        return s + ')'

    def __len__(self):
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self):
        # infix a default print
        return self.infix()
    
    def val(self, series, deterministic=False):
        
        curr_dim = len(series) %self.params.dimension
        if len(self.children) == 0:
            if str(self.value).startswith('x_'):
                _, dim, offset = self.value.split('_')
                dim, offset = int(dim), int(offset)
                dim_offset = dim-curr_dim
                return series[-offset*self.params.dimension+dim_offset]
            elif str(self.value) == 'n':
                return len(series)
            elif str(self.value) == 'rand':
                if deterministic: return 0
                if self.params.real_series:
                    return np.random.randn()
                else:
                    return np.random.choice([-1,0,1])
            elif str(self.value) in math_constants:
                return getattr(np, str(self.value))
            else:
                return int(self.value)
        
        if self.value == 'add':
            return self.children[0].val(series) + self.children[1].val(series)
        if self.value == 'sub':
            return self.children[0].val(series) - self.children[1].val(series)
        if self.value == 'mul':
            return self.children[0].val(series) * self.children[1].val(series)
        if self.value == 'pow':
            return self.children[0].val(series) ** self.children[1].val(series)
        if self.value == 'max':
            return max(self.children[0].val(series), self.children[1].val(series))
        if self.value == 'min':
            return min(self.children[0].val(series), self.children[1].val(series))
        if self.value == 'mod':
            if self.children[1].val(series)==0: return np.nan
            else: return self.children[0].val(series) % self.children[1].val(series) 
        if self.value == 'div':
            if self.children[1].val(series)==0: return np.nan
            else: return self.children[0].val(series) / self.children[1].val(series)
        if self.value == 'idiv':
            if self.children[1].val(series)==0: return np.nan
            else: return self.children[0].val(series) // self.children[1].val(series)
        if self.value == 'inv':
            return 1/(self.children[0].val(series))
        if self.value == 'sqr':
            return (self.children[0].val(series))**2
        if self.value == 'abs':
            return abs(self.children[0].val(series))
        else:
            return getattr(np,self.value)(self.children[0].val(series))
        

        
class NodeList():
    def __init__(self, nodes):
        self.nodes = []
        for node in nodes:
            self.nodes.append(node)
        self.params = nodes[0].params   
        
    def infix(self):
        return ' | '.join([node.infix() for node in self.nodes])
    
    def __len__(self):
        return sum([len(node) for node in self.nodes])
    
    def prefix(self):
        return ',|,'.join([node.prefix() for node in self.nodes])
    
    def __str__(self):
        return self.infix()
    
    def val(self, series, deterministic=False):
        return [node.val(series, deterministic=deterministic) for node in self.nodes]
    

    
class Generator(ABC):
    def __init__(self, params):
        pass
        
    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass


    
class RandomRecurrence(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        
        self.real_series = params.real_series
        self.prob_const = params.prob_const
        self.prob_n = params.prob_n
        self.prob_rand = params.prob_rand
        self.max_int = params.max_int
        self.max_degree = params.max_degree
        self.max_ops = params.max_ops
        self.max_len = params.max_len
        self.init_scale = params.init_scale
        self.dimension = params.dimension
        
        if params.real_series:
            self.max_number = 10**(params.max_exponent+params.float_precision)
            self.operators = operators_real
        else:
            self.max_number = params.max_number
            self.operators = operators_int
        self.unaries = [o for o in self.operators.keys() if self.operators[o] == 1]
        self.binaries = [o for o in self.operators.keys() if self.operators[o] == 2]
        self.unary = len(self.unaries) > 0
        self.distrib = self.generate_dist(2 * self.max_ops)

        self.constants = [str(i) for i in range(-self.max_int,self.max_int+1) if i!=0]
        if params.real_series:
            self.constants += math_constants
        self.symbols = list(self.operators) + [f'x_{i}_{j}' for i in range(self.dimension) for j in range(self.max_degree+1)] + self.constants + ['n', '|']
        if params.prob_rand:
            self.symbols += ['rand']

    def generate_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n- 1, e) + D(n - 1, e + 1)
        p1 =  if binary trees, 1 if unary binary
        """
        p1 = 1 if self.unary else 0
        # enumerate possible trees
        D = []
        D.append([0] + ([1 for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(s[e - 1] + p1 * D[n - 1][e] + D[n - 1][e + 1])
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        return D

    def generate_leaf(self, degree):
        draw = self.rng.rand()
        if draw < self.prob_const:
            return self.rng.choice(self.constants)
        elif draw > self.prob_const and draw < self.prob_const + self.prob_n:
            return 'n'
        elif draw > self.prob_const + self.prob_n and draw < self.prob_const + self.prob_n + self.prob_rand:
            return 'rand'
        else:
            return f'x_{self.rng.randint(self.dimension)}_{self.rng.randint(degree)+1}'

    def generate_ops(self, arity):
        if arity==1:
            ops = [unary for unary in self.unaries]
        else:
            ops = [binary for binary in self.binaries]
        return self.rng.choice(ops)

    def sample_next_pos(self, nb_empty, nb_ops):
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `nb_empty` - 1}.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        if self.unary:
            for i in range(nb_empty):
                probs.append(self.distrib[nb_ops - 1][nb_empty - i])
        for i in range(nb_empty):
            probs.append(self.distrib[nb_ops - 1][nb_empty - i + 1])
        probs = [p / self.distrib[nb_ops][nb_empty] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = self.rng.choice(len(probs), p=probs)
        arity = 1 if self.unary and e < nb_empty else 2
        e %= nb_empty
        return e, arity

    def generate_tree(self, nb_ops, degree):
        tree = Node(0, self.params)
        empty_nodes = [tree]
        next_en = 0
        nb_empty = 1
        while nb_ops > 0:
            next_pos, arity = self.sample_next_pos(nb_empty, nb_ops)
            for n in empty_nodes[next_en:next_en + next_pos]:
                n.value = self.generate_leaf(degree)
            next_en += next_pos
            op = self.generate_ops(arity)
            empty_nodes[next_en].value = op
            for _ in range(arity):
                e = Node(0, self.params)
                empty_nodes[next_en].push_child(e)
                empty_nodes.append(e)
            nb_empty += arity - 1 - next_pos
            nb_ops -= 1
            next_en += 1
        for n in empty_nodes[next_en:]:
            n.value = self.generate_leaf(degree)
        
        #tree = self.check_tree(tree, degree)
        
        return tree
    
    #def check_tree(self, node, degree):
    #    '''
    #    Remove identical leafs
    #    '''
    #    if len(node.children)==0: return node
    #    elif len(node.children)==1: 
    #        if node.children[0].children:
    #            return self.check_tree(node.children[0], degree)
    #        else: 
    #            while isinstance(node.children[0].value,int):
    #                node.children[0].value = self.generate_leaf(degree)
    #    else:
    #        node.children[0] = self.check_tree(node.children[0], degree)
    #        node.children[1] = self.check_tree(node.children[1], degree)
    #        if bool(node.children[0].children or node.children[1].children): return node
    #        while (node.children[0].value == node.children[1].value) or (isinstance(node.children[0].value,int) and isinstance(node.children[1].value,int)):
    #            node.children[1].value = self.generate_leaf(degree)
    #    return node
        
    def generate(self, rng, nb_ops=None, deg=None, length=None):
        self.rng = rng
        self.rng.seed() # TODO : fix this

        if deg is None:    deg    = self.rng.randint(1, self.max_degree + 1)
        if length is None: length = self.rng.randint(3*deg, self.max_len+1)

        trees = []
        if nb_ops is None: nb_ops = self.rng.randint(1, self.max_ops + 1, size=(self.dimension,))
        elif type(nb_ops)==int: nb_ops = [nb_ops]*self.dimension
            
        for i in range(self.dimension):
            trees.append(self.generate_tree(nb_ops[i],deg))
        tree = NodeList(trees)
        
        series = []
        for i in range(deg*self.dimension):
            if self.real_series: 
                series.append(self.rng.uniform(-self.init_scale, self.init_scale))
            else:
                series.append(self.rng.randint(-self.init_scale, self.init_scale+1))
        for i in range(deg, length):
            vals = tree.val(series)
            if any([np.isnan(x) for x in vals]) or any([abs(x)>self.max_number for x in vals]): 
                return None, None
            series.extend(vals)
        return tree, series


    def evaluate(self, src, tgt, hyp, n_predictions=3):
        errors = []
        for i in range(n_predictions):
            try:
                pred = hyp.val(src, deterministic=True)
                true = tgt.val(src, deterministic=True)
                src.extend(true)
                errors.append(max([abs(float(p-t)/float(t+1e-100)) for p,t in zip(pred, true)]))
            except Exception as e:
                print(e)
                return -1
        return max(errors)        

    def evaluate_without_target(self, src, hyp, n_predictions=3):
        errors = []
        targets = src[-n:]
        src = src[:-n]
        for i in range(n_predictions):
            pred = hyp.val(src)
            true = targets[i]
            src.extend(true)
            errors.append(max([abs(float(p-t)/float(t+1e-100)) for p,t in zip(pred, true)]))
            # except:
            #     return -1
        return max(errors)
        



