# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
A script to run multinode training with submitit.
"""

import argparse
import os
import uuid
from pathlib import Path
import time
import shutil
from distutils import dir_util
import itertools

import train as classification
import submitit

def parse_args():
    classification_parser = classification.get_parser()
    parser = argparse.ArgumentParser("Submitit for recur", parents=[classification_parser])
    parser.add_argument("--ngpus", default=0, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--ncpus", default=60, type=int, help="Number of cpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=1000, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="devlab,learnfair,scavenge", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="icml", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/recur")
        p = p / "data"
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import train as classification

        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        checkpoint_file = os.path.join(self.args.exp_name, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.load_checkpoint = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
            
def main():
    
    args = parse_args()
    shared_folder = get_shared_folder()
    for f in os.listdir():
        if f.endswith('.py'):
            shutil.copy2(f, shared_folder)
    # if not os.path.exists(os.path.join(shared_folder,'src')):
    dir_util.copy_tree('src', os.path.join(shared_folder,'src'))
    os.chdir(shared_folder)

    grid = {
        'env_base_seed':[-1]*2
    }

    def dict_product(d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))

    for params in dict_product(grid):

        args.cpu = True
        args.export_data = True

        args.exp_name = shared_folder.name
        args.dump_path = shared_folder.parent
        
        # Note that the folder will depend on the job_id, to easily track experiments
        executor = submitit.AutoExecutor(folder=args.exp_name, slurm_max_num_timeout=30)

        kwargs = {}
        kwargs['slurm_constraint'] = 'pascal'
        if args.comment:
            kwargs['slurm_comment'] = args.comment
        executor.update_parameters(
            nodes=args.nodes,
            tasks_per_node=1,
            gpus_per_node=args.ngpus,
            cpus_per_task=args.ncpus,
            timeout_min=args.timeout,  # max is 60 * 72
            slurm_partition=args.partition,
            slurm_signal_delay_s=120,
            **kwargs
        )

        executor.update_parameters(name="generate")

        for k,v in params.items():
            setattr(args,k,v)

        trainer = Trainer(args)
        job = executor.submit(trainer)

        print("Submitted job_id:", job.job_id)

if __name__ == "__main__":
    main()
