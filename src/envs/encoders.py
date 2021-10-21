from abc import ABC, abstractmethod
import numpy as np
import math
from .generators import all_operators, Node, NodeList

class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self, params):
        pass

    @abstractmethod
    def encode(self, val):
        pass

    @abstractmethod
    def decode(self, lst):
        pass

class IntegerSeries(Encoder):
    def __init__(self, params):
        super().__init__(params)
        self.int_base = params.int_base
        self.symbols = ['+', '-', ',']
        self.symbols.extend([str(i) for i in range(self.int_base)])

    def encode(self, value):
        seq = []
        for val in value:
            seq.append('+' if val >= 0 else '-')
            vseq = []
            w = abs(val)
            if w == 0:
                seq.append('0')
            else:
                while w > 0:
                    vseq.append(str(w % self.int_base))
                    w = w // self.int_base
                seq.extend(vseq[::-1])
        return seq

    def decode(self, lst):
        
        if len(lst) == 0:
            return None
        res = []
        if lst[0] in ["+", "-"]:
            curr_group = [lst[0]]
        else:
            return None
        if lst[-1] in ["+", "-"]:
            return None

        for x in lst[1:]:
            if x in ["+", "-"]:
                if len(curr_group)>1:
                    sign = 1 if curr_group[0]=="+" else -1
                    value = 0
                    for elem in curr_group[1:]:
                        value = value*self.int_base + int(elem)
                    res.append(sign*value)
                    curr_group = [x]
                else:
                    return None
            else:
                curr_group.append(x)
        if len(curr_group)>1:
            sign = 1 if curr_group[0]=="+" else -1
            value = 0
            for elem in curr_group[1:]:
                value = value*self.int_base + int(elem)
            res.append(sign*value)
        return res
    
class RealSeries(Encoder):
    def __init__(self, params):
        super().__init__(params)
        self.float_precision = params.float_precision
        self.max_exponent = params.max_exponent
        self.max_token = 10 ** (self.float_precision + 1)
        self.symbols = ['+','-']
        self.symbols.extend(['N' + str(i) for i in range(self.max_token)])
        self.symbols.extend(['E' + str(i) for i in range(-self.max_exponent, self.max_exponent+1)])

    def encode(self, value):
        """
        Write a float number
        """
        seq = []
        for val in value:
            precision = self.float_precision
            assert val not in [-np.inf, np.inf]
            sign = '+' if val>=0 else '-'
            m, e = (f"%.{precision}e" % val).split("e")
            i, f = m.split(".")
            i = i + f
            ipart = abs(int(i))
            expon = int(e) - precision
            if expon < -self.max_exponent:
                ipart = 0
            if ipart == 0:
                expon = 0
            seq.extend([sign, 'N' + str(ipart), "E" + str(expon)])
        return seq
    
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def decode(self, lst):
        """
        Parse a list that starts with a float.
        Return the float value, and the position it ends in the list.
    """
        if len(lst)==0:
            return None
        seq = []
        for val in self.chunks(lst, 3):
            for x in val: 
                if x[0] not in ['-','+','E','N']: return np.nan
            try:
                sign = 1 if val[0]=='+' else -1
                mant = int(val[1][1:])
                exp = int(val[2][1:])
                value = sign * mant * (10 ** exp)
                value=float(value)
            except Exception:
                value = np.nan
            seq.append(value)
        return seq
    
class Equation(Encoder):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        #self.int_encoder = IntegerSeries(params)

    def encode(self, tree):
        pref = tree.prefix().split(',')
        res = []
        for p in pref:
            res.append(p)
        return res
    
    def _decode(self, lst):
        if len(lst)==0:
            return None, 0
        elif lst[0]=='SPECIAL':
            return None, 0
        elif lst[0] in all_operators.keys():
            res = Node(lst[0], self.params)
            arity = all_operators[lst[0]]
            pos = 1
            for i in range(arity):
                child, length = self._decode(lst[pos:])
                if child is None:
                    return None, pos
                res.push_child(child)
                pos += length
            return res, pos
        else:
            return Node(lst[0], self.params), 1        

    def split_at_value(self, lst, value):
        indices = [i for i, x in enumerate(lst) if x==value]
        res = []
        for start, end in zip([0, *[i+1 for i in indices]], [*[i-1 for i in indices], len(lst)]):
            res.append(lst[start:end+1])
        return res
    
    def decode(self, lst):
        trees = []
        lists = self.split_at_value(lst, '|')
        for lst in lists:
            tree = self._decode(lst)[0]
            if tree is None: return None
            trees.append(tree)
        tree = NodeList(trees)
        return tree
            
