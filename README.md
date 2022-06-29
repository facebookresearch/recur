# Deep Symbolic Regression for Recurrent Sequences

This repository contains code for the paper Deep Symbolic Regression for Recurrent Sequences.
The code is based on the repository [Deep Learning for Symbolic Mathematics](https://github.com/facebookresearch/SymbolicMathematics).
Most of the code specific to recurrent sequences lies in the folder ```src/envs```.

## Run the model

To launch a small Transformer on the CPU, run:
```python train.py --cpu True```

## Main arguments

--float_sequences		# if True, run the float model, otherwise run the integer model
--output_numeric		# if True, run the numeric model, otherwise run the symbolic model
--use_sympy 			# whether to use sympy simplification

The arguments specific to the generator can be found in ```src/envs/recurrence.py```.

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [SymPy](https://www.sympy.org/)
- [PyTorch](http://pytorch.org/) (tested on version 1.3)
