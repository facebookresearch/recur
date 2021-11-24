import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

class Simplifier():
    
    def __init__(self, encoder, generator):
        
        self.encoder = encoder
        self.params = generator.params
        self.operators = generator.operators
        self.max_int = generator.max_int
        self.local_dict = {k: sp.Symbol(k, real=True, nonzero=True) for k in (generator.variables+generator.math_constants)}

    def simplify_tree(self, tree):
        prefix = tree.prefix().split(',')
        simplified_prefix = self.simplify_prefix(prefix)
        simplified_tree = self.encoder.decode(simplified_prefix)
        return simplified_tree
    
    def simplify_prefix(self, prefix):
        infix = self.prefix_to_infix(prefix)
        return self.sympy_to_prefix(self.infix_to_sympy(infix, simplify = self.params.simplify))
    
    def get_simple_infix(self, tree):
        infix = self.prefix_to_infix(tree.prefix().split(','))
        return self.infix_to_sympy(infix, simplify = self.params.simplify)
        
    def write_infix(self, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == 'add':
            return f'({args[0]})+({args[1]})'
        elif token == 'sub':
            return f'({args[0]})-({args[1]})'
        elif token == 'mul':
            return f'({args[0]})*({args[1]})'
        elif token == 'div':
            return f'({args[0]})/({args[1]})'
        elif token == 'idiv':
            return f'idiv({args[0]},{args[1]})'
        elif token == 'mod':
            return f'({args[0]})%({args[1]})'
        elif token == 'abs':
            return f'Abs({args[0]})'
        elif token == 'inv':
            return f'1/({args[0]})'
        elif token == 'sqr':
            return f'({args[0]})**2'
        elif token in self.operators:
            return f'{token}({args[0]})'
        else:
            return token
        raise InvalidPrefixExpression(f"Unknown token in prefix expression: {token}, with arguments {args}")

    def _prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.operators[t]):
                i1, l1 = self._prefix_to_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        else: # leaf
            return t, expr[1:]

    def prefix_to_infix(self, expr):
        """
        Prefix to infix conversion.
        """
        p, r = self._prefix_to_infix(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed.")
        return f'({p})'


    def infix_to_sympy(self, infix, simplify=False):
        """
        Convert an infix expression to SymPy.
        """
        expr = parse_expr(infix, evaluate=True, local_dict=self.local_dict)
        if simplify: expr = sp.simplify(expr)
        return expr
    
    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        assert (op == 'add' or op == 'mul') and (n_args >= 2) or (op != 'add' and op != 'mul') and (1 <= n_args <= 2)

        # square root
        # if op == 'pow':
        #     if isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
        #         return ['sqrt'] + self.sympy_to_prefix(expr.args[0])
        #     elif str(expr.args[1])=='2':
        #         return ['sqr'] + self.sympy_to_prefix(expr.args[0])
        #     elif str(expr.args[1])=='-1':
        #         return ['inv'] + self.sympy_to_prefix(expr.args[0])
        #     elif str(expr.args[1])=='-2':
        #         return ['inv', 'sqr'] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return [str(expr)]
        elif isinstance(expr, sp.Rational):
            return ['mul',str(expr.p),'pow',str(expr.q),'-1']
        elif expr == sp.E:
            return ['e']
        elif expr == sp.pi:
            return ['pi']
        
        # if we want div and sub
        #if isinstance(expr, sp.Mul) and len(expr.args)==2:
        #    if isinstance(expr.args[0], sp.Mul) and isinstance(expr.args[0].args[0], sp.Pow): return ['div']+self.sympy_to_prefix(expr.args[1])+self.sympy_to_prefix(expr.args[0].args[1])
        #    if isinstance(expr.args[1], sp.Mul) and isinstance(expr.args[1].args[0], sp.Pow): return ['div']+self.sympy_to_prefix(expr.args[0])+self.sympy_to_prefix(expr.args[1].args[1])
        #if isinstance(expr, sp.Add) and len(expr.args)==2:
        #    if isinstance(expr.args[0], sp.Mul) and str(expr.args[0].args[0])=='-1': return ['sub']+self.sympy_to_prefix(expr.args[1])+self.sympy_to_prefix(expr.args[0].args[1])
        #    if isinstance(expr.args[1], sp.Mul) and str(expr.args[1].args[0])=='-1': return ['sub']+self.sympy_to_prefix(expr.args[0])+self.sympy_to_prefix(expr.args[1].args[1])
        
        # if isinstance(expr, sp.Pow) and str(expr.args[1])=='-1':
        #     return ['inv'] + self.sympy_to_prefix(expr.args[0])
        
        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
            
        # Unknown operator
        return self._sympy_to_prefix(str(type(expr)), expr)     
      
        
    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: 'add',
        sp.Mul: 'mul',
        sp.Mod: 'mod',
        sp.Abs: 'abs',
        sp.Pow: 'pow',
        sp.sign: 'sign',
        # Exp functions
        sp.exp: 'exp',
        sp.log: 'log',
        # Trigonometric Functions
        sp.sin: 'sin',
        sp.cos: 'cos',
        sp.tan: 'tan',
        # Trigonometric Inverses
        sp.asin: 'asin',
        sp.acos: 'acos',
        sp.atan: 'atan',
    }
