# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import numpy as np

from .utils import to_cuda


TOLERANCE_THRESHOLD = 1e-1


logger = getLogger()


def idx_to_infix(env, idx, input=True):
    """
    Convert an indexed prefix expression to SymPy.
    """
    if input:
        prefix = [env.input_id2word[wid] for wid in idx]
        infix = env.input_to_infix(prefix)
    else:
        prefix = [env.output_id2word[wid] for wid in idx]
        infix = env.output_to_infix(prefix)
    return infix

def check_hypothesis(eq):
    """
    Check a hypothesis for a given equation and its solution.
    """
    env = Evaluator.ENV
    n = env.params.n_predictions

    src = [env.input_id2word[wid] for wid in eq["src"]]
    tgt = [env.output_id2word[wid] for wid in eq["tgt"]]
    hyp = [env.output_id2word[wid] for wid in eq["hyp"]]

    # update hypothesis

    eq["src"] = env.input_to_infix(src)
    eq["tgt"] = tgt
    eq["hyp"] = hyp
    
    error = env.check_prediction(src, tgt, hyp, n)
    eq["error"] = error

    return eq

def test_fn():
    print("shiit")

class Evaluator(object):

    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env

    def run_all_evals(self):
        """
        Run all evaluations.

        """
        params = self.params
        scores = OrderedDict({"epoch": self.trainer.epoch})

        # save statistics about generated data
        if params.export_data:
            scores["total"] = self.trainer.total_samples
            return scores

        with torch.no_grad():
            for data_type in ["valid"]:
                for task in params.tasks:
                    if params.beam_eval:
                        self.enc_dec_step_beam(data_type, task, scores)
                    else:
                        self.enc_dec_step(data_type, task, scores)
        return scores

    def enc_dec_step(self, data_type, task, scores):
        """
        Encoding / decoding step.
        """
        params = self.params
        env = self.env
        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in ["recurrence"]

        # stats
        xe_loss = 0
        n_valid = torch.zeros(10000, dtype=torch.long)
        n_total = torch.zeros(10000, dtype=torch.long)

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(
                params.dump_path, f"eval.{data_type}.{task}.{scores['epoch']}"
            )
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        # iterator
        iterator = self.env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=params.eval_size,
            input_length_modulo=params.eval_ablation_input_length
        )
        
        eval_size = len(iterator.dataset)
        logger.info('Testing on {} samples'.format(eval_size))

        for (x1, len1), (x2, len2), nb_ops in iterator:
            
            # print status
            # FC : remove lengthy pacifiers
            # if n_total.sum().item() % 500 < params.batch_size_eval:
            #    logger.info(f"{n_total.sum().item()}/{eval_size}")

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                alen[:, None] < len2[None] - 1
            )  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1_, len1_, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

            # forward / loss
            encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1_,
            )
            word_scores, loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
            )

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()

            # export evaluation details
            if params.eval_verbose:
                for i in range(len(len1)):
                    src = idx_to_infix(env, x1[1 : len1[i] - 1, i].tolist(), True)
                    tgt = idx_to_infix(env, x2[1 : len2[i] - 1, i].tolist(), False)
                    s = f"Equation {n_total.sum().item() + i} "
                    s += f"({'Valid' if valid[i] else 'Invalid'})\n"
                    s += f"src={src}\ntgt={tgt}"
                    if params.eval_verbose_print:
                        logger.info(s)
                    f_export.write(s + "\n")
                    f_export.flush()

            # stats
            xe_loss += loss.item() * len(y)
            n_valid.index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

        # evaluation details
        if params.eval_verbose:
            f_export.close()

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(
            f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) "
            f"equations were evaluated correctly."
        )

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size
        scores[f"{data_type}_{task}_xe_loss"] = xe_loss / _n_total
        scores[f"{data_type}_{task}_acc"] = 100.0 * _n_valid / _n_total

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            e = env.decode_class(i)
            scores[f"{data_type}_{task}_acc_{e}"] = (
                100.0 * n_valid[i].item() / max(n_total[i].item(), 1)
            )
            if n_valid[i].item() > 0:
                logger.info(
                    f"{e}: {n_valid[i].item()} / {n_total[i].item()} "
                    f"({100. * n_valid[i].item() / max(n_total[i].item(), 1)}%)"
                )

    def enc_dec_step_beam(self, data_type, task, scores, size=None):
        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        params = self.params
        env = self.env
        max_beam_length = self.params.max_output_len
        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1, 2]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in ["recurrence"]

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(
                params.dump_path, f"eval.beam.{data_type}.{task}.{scores['epoch']}"
            )
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        def display_logs(logs, offset):
            """
            Display detailed results about success / fails.
            """
            if params.eval_verbose == 0:
                return
            for i, res in sorted(logs.items()):
                n_valid = sum([int(v) for _, _, v in res["hyps"]])
                s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\n"
                s += f"src={res['src']}\ntgt={res['tgt']}\n"
                for hyp, score, valid in res["hyps"]:
                    validity = 'Valid' if valid else 'Invalid'
                    if score is None:
                        s += f"{validity} {hyp}\n"
                    else:
                        s += f"{validity} {score :.3e} {hyp}\n"
                if params.eval_verbose_print:
                    logger.info(s)
                f_export.write(s + "\n")
                f_export.flush()

        # iterator
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=params.eval_size,
            input_length_modulo=params.eval_input_length_modulo
        )
        eval_size = len(iterator.dataset)

        # stats
        xe_loss = 0
        n_perfect_match = 0
        n_correct = 0
        n_valid_per_info,n_total_per_info = None,None
        n_valid_additional = torch.zeros(1+len(env.additional_tolerance))
        n_valid_per_n_predictions= torch.zeros(params.n_predictions,dtype=torch.long)

        for (x1, len1), (x2, len2), infos in iterator:
            if n_valid_per_info is None:
                info_types=list(infos.keys()) 
                first_key=info_types[0]
                n_valid_per_info={info_type: torch.zeros(10000, dtype=torch.long) for info_type in info_types}
                n_total_per_info={info_type: torch.zeros(10000, dtype=torch.long) for info_type in info_types}



            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                alen[:, None] < len2[None] - 1
            )  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1_, len1_, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
            bs = len(len1)

            valid_additional_per_minibatch = torch.zeros(1+len(env.additional_tolerance), bs, dtype=int)
            valid_per_info_type_per_minibatch={info_type: torch.zeros(10000, bs, dtype=torch.long) for info_type in info_types}
            valid_per_n_predictions_per_minibatch=torch.zeros(params.n_predictions, bs, dtype=torch.long)

            # forward
            encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1_,
            )
            word_scores, loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
            )

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()
            n_perfect_match += valid.sum().item()

            # save evaluation details
            beam_log = {}
            for i in range(len(len1)):
                src = idx_to_infix(env, x1[1 : len1[i] - 1, i].tolist(), True)
                tgt = idx_to_infix(env, x2[1 : len2[i] - 1, i].tolist(), False)
                if valid[i]:
                    beam_log[i] = {"src": src, "tgt": tgt, "hyps": [(tgt, None, True)]}

            # stats
            xe_loss += loss.item() * len(y)
            for info_type in infos.keys():
                valid_per_info_type_per_minibatch[info_type].index_add_(0, infos[info_type], valid)
                n_total_per_info[info_type].index_add_(0, infos[info_type], torch.ones_like(infos[info_type]))

            # continue if everything is correct. if eval_verbose, perform
            # a full beam search, even on correct greedy generations
            if valid.sum() == len(valid) and params.eval_verbose < 2:
                display_logs(beam_log, offset=n_total_per_info[first_key].sum().item() - bs)
                continue

            # invalid top-1 predictions - check if there is a solution in the beam
            invalid_idx = (1 - valid).nonzero().view(-1)
            logger.info(
                f"({n_total_per_info[first_key].sum().item()}/{eval_size}) Found "
                f"{bs - len(invalid_idx)}/{bs} valid top-1 predictions. "
                f"Generating solutions ..."
            )

            # generate
            _, _, generations = decoder.generate_beam(
                encoded.transpose(0, 1),
                len1_,
                beam_size=params.beam_size,
                length_penalty=params.beam_length_penalty,
                early_stopping=params.beam_early_stopping,
                max_len=max_beam_length,
            )

            # prepare inputs / hypotheses to check
            # if eval_verbose < 2, no beam search on equations solved greedily
            inputs = []
            for i in range(len(generations)):
                if valid[i] and params.eval_verbose < 2:
                    continue
                for j, (score, hyp) in enumerate(
                    sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)
                ):
                    inputs.append(
                        {
                            "i": i,
                            "j": j,
                            "score": score,
                            "src": x1[1 : len1[i] - 1, i].tolist(),
                            "tgt": x2[1 : len2[i] - 1, i].tolist(),
                            "hyp": hyp[1:].tolist(),
                            "task": task,
                        }
                    )

            # check hypotheses with multiprocessing
            outputs = []
            if params.windows is True:
                for inp in inputs:
                    outputs.append(check_hypothesis(inp))
            else:
                with ProcessPoolExecutor(max_workers=20) as executor:
                    for output in executor.map(check_hypothesis, inputs, chunksize=1):
                        outputs.append(output)


            correct = torch.zeros(bs, dtype=int)
            # read results
            for i in range(bs):

                # select hypotheses associated to current equation
                gens = sorted([o for o in outputs if o["i"] == i], key=lambda x: x["j"])
                assert (len(gens) == 0) == (valid[i] and params.eval_verbose < 2) and (
                    i in beam_log
                ) == valid[i]
                if len(gens) == 0:
                    continue

                # source / target
                src = gens[0]["src"]
                tgt = gens[0]["tgt"]
                beam_log[i] = {"src": src, "tgt": tgt, "hyps": []}

                # for each hypothesis
                for j, gen in enumerate(gens):

                    # sanity check
                    assert (
                        gen["src"] == src
                        and gen["tgt"] == tgt
                        and gen["i"] == i
                        and gen["j"] == j
                    )

                    # if hypothesis is correct, and we did not find a correct one before
                    error = gen["error"]
                    error_infty = max(error)
                    is_valid = error_infty >= 0.0 and error_infty < env.float_tolerance
                    if error_infty >= 0.0 and not valid[i]:
                        correct[i] = 1
                        for k, tol in enumerate(env.additional_tolerance):
                            if error_infty < tol:
                                valid_additional_per_minibatch[k,i] = 1
                        for k in range(params.n_predictions):
                            valid_per_n_predictions_per_minibatch[k,i]=int(max(error[:k+1])<env.float_tolerance)
                        if error_infty < env.float_tolerance:
                            for info_type in infos.keys():
                                valid_per_info_type_per_minibatch[info_type][infos[info_type][i],i]= 1
                            valid[i] = 1
                    # update beam log
                    beam_log[i]["hyps"].append((gen["hyp"], gen["score"], is_valid))

            n_correct += correct.sum().item()
            n_valid_additional += valid_additional_per_minibatch.sum(1)
            n_valid_per_n_predictions += valid_per_n_predictions_per_minibatch.sum(1)
            for info_type in info_types:
                n_valid_per_info[info_type] += valid_per_info_type_per_minibatch[info_type].sum(1)

            # valid solutions found with beam search
            logger.info(
                f"    Found {valid.sum().item()}/{bs} solutions in beam hypotheses."
            )

            # export evaluation details
            if params.eval_verbose:
                assert len(beam_log) == bs
                display_logs(beam_log, offset=n_total_per_info[first_key].sum().item() - bs)

        # evaluation details
        if params.eval_verbose:
            f_export.close()
            logger.info(f"Evaluation results written in {eval_path}")

        # log
        n_valid = n_valid_per_info[first_key]
        n_total = n_total_per_info[first_key]
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(
            f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) "
            f"equations were evaluated correctly."
        )

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size
        scores[f"{data_type}_{task}_xe_loss"] = xe_loss / _n_total
        scores[f"{data_type}_{task}_perfect"] = 100.0 * n_perfect_match / _n_total
        scores[f"{data_type}_{task}_correct"] = 100.0 * (n_perfect_match + n_correct) / _n_total
        scores[f"{data_type}_{task}_beam_acc"]= 100.0 * _n_valid / _n_total
        
        for i in range(len(env.additional_tolerance)):
            scores[f"{data_type}_{task}_additional_{i+1}"] = (
                100.0 * (n_perfect_match + n_valid_additional[i].item()) / _n_total
            )
        for i in range(params.n_predictions):
            scores[f"{data_type}_{task}_n_predictions_{i+1}"] = (
                100.0 * (n_perfect_match + n_valid_per_n_predictions[i].item()) / _n_total
            )

        # per class perplexity and prediction accuracy
        for info_type in info_types:
            n_valid = n_valid_per_info[info_type]
            n_total = n_total_per_info[info_type]
            for i in range(len(n_total)):
                if n_total[i].item() == 0:
                    continue
                e = env.decode_class(i)
                logger.info(
                    f"{e} operators: {n_valid[i].sum().item()} / {n_total[i].item()} "
                    f"({100. * n_valid[i].sum().item() / max(n_total[i].item(), 1)}%)"
                )
                scores[f"{data_type}_{task}_beam_acc_{info_type}_{e}"] = (
                    100.0 * n_valid[i].sum().item() / max(n_total[i].item(), 1)
                )
