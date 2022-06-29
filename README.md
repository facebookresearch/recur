# Deep Symbolic Regression for Recurrent Sequences

This repository contains code for the paper [Deep Symbolic Regression for Recurrent Sequences](https://arxiv.org/abs/2201.04600).
An interactive demonstration of the paper may be found [here](https://symbolicregression.metademolab.com/).

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

## Multinode training

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):
```
pip install submitit
```

To launch a run on 2 nodes with 8 GPU each, use the ```run_with_submitit``` script.

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [SymPy](https://www.sympy.org/)
- [PyTorch](http://pytorch.org/) (tested on version 1.3)

## Citation

If you want to reuse this material, please considering citing the following:
```
@article{d2022deep,
  title={Deep symbolic regression for recurrent sequences},
  author={d'Ascoli, St{\'e}phane and Kamienny, Pierre-Alexandre and Lample, Guillaume and Charton, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:2201.04600},
  year={2022}
}
```

## License

The majority of this repository is released under the Apache 2.0 license as found in the LICENSE file.