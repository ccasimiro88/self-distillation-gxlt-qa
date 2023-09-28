# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" This is the almost the same script as `examples/run_squad_w_distillation.py` with an additional
option for parallel question format to perform class-wise distillation."""

from processor_squad_multilingual import (
    SquadResult,
    SquadV1Processor,
    SquadV2Processor,
    SquadProcessorMultilingual,
)
from processor_squad_multilingual import (
    squad_convert_examples_to_features as squad_convert_examples_to_features_parallel,
)
import argparse
import glob
import logging
import os
import random
import timeit
import sys
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils import (
    kl_divergence,
    compute_kl_map_at_k_coefficients,
    compute_kl_mrr_at_k_coefficients,
)

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    squad_convert_examples_to_features,
    AutoTokenizer,
    AutoConfig,
    AutoModelForQuestionAnswering,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

# Import class-wise specific classes and functions
sys.path.append(os.path.dirname(__file__))


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (
        DistilBertConfig,
        DistilBertForQuestionAnswering,
        DistilBertTokenizer,
    ),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


# Mean Pooling - Take attention mask into account for correct averaging
# from https://huggingface.co/sentence-transformers/stsb-roberta-large


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def compute_mse_loss(emedding_stu, emedding_tea):
    mse = nn.MSELoss()
    loss = mse(emedding_stu, emedding_tea)

    return loss


def compute_distillation_loss(
    args,
    start_positions,
    end_positions,
    start_logits_stu,
    start_logits_tea,
    end_logits_stu,
    end_logits_tea,
    type,
):
    loss_start = kl_divergence(
        type, start_logits_stu, start_logits_tea, args.temperature
    )
    loss_end = kl_divergence(type, end_logits_stu, end_logits_tea, args.temperature)

    loss_kl = (loss_start + loss_end) / 2.0

    if args.use_map_loss_coefficients:
        alpha_kl = compute_kl_map_at_k_coefficients(
            start_positions,
            end_positions,
            start_logits_tea,
            end_logits_tea,
            args.temperature,
        )
    elif args.use_mrr_loss_coefficients:
        alpha_kl = compute_kl_mrr_at_k_coefficients(
            start_positions,
            end_positions,
            start_logits_tea,
            end_logits_tea,
            args.temperature,
        )
    else:
        alpha_kl = 1.0
    return loss_kl, alpha_kl


def train(args, train_dataset, model, tokenizer, teacher=None):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tensorboard_dir = args.logging_dir if args.logging_dir else args.output_dir
        tb_writer = SummaryWriter(tensorboard_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        if args.linear_schedule_epoch:
            len(train_dataloader)
        else:
            t_total = (
                len(train_dataloader)
                // args.gradient_accumulation_steps
                * args.num_train_epochs
            )

    # Freeze the base model params
    if args.train_layers:
        for name, param in model.named_parameters():
            if all([bool(name.find(layer) == -1) for layer in args.train_layers]):
                param.requires_grad = False
            else:
                logging.info(f"Training layer: {name}")

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    if args.constant_lr:
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info(
                "  Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch,
            )
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    best_validation_score = 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    # Added here for reproductibility
    set_seed(args)

    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        epoch_steps = len(epoch_iterator) // args.gradient_accumulation_steps

        # Set the teacher model equal to the student to apply self-distillation at each epoch
        if args.self_distil and epoch > 0:
            teacher = copy.deepcopy(model)

        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            if teacher is not None:
                teacher.eval()
            # split the batch for the teacher and the student model using the second dim of the tensors
            # of shape [num_examples, num_parallel_examples, max_seq_len]
            if all(b.shape[1] == 2 for b in batch):
                batch_qa = [b[:, 0] for b in batch]
                batch_parallel_qa = [b[:, 1] for b in batch]

            # Squeeze the extra "num_parallel_examples" dimension (dim 1) and
            # go back to the standard case with one input
            else:
                batch_parallel_qa = batch_qa = [b.squeeze(1) for b in batch]

            batch_parallel_qa = tuple(t.to(args.device) for t in batch_parallel_qa)
            batch_qa = tuple(t.to(args.device) for t in batch_qa)
            inputs_student = {
                "input_ids": batch_qa[0],
                "attention_mask": batch_qa[1],
                "start_positions": batch_qa[3],
                "end_positions": batch_qa[4],
            }
            if args.model_type != "distilbert":
                inputs_student["token_type_ids"] = (
                    None if args.model_type == "xlm" else batch_qa[2]
                )
            if args.model_type in ["xlnet", "xlm"]:
                inputs_student.update({"cls_index": batch_qa[5], "p_mask": batch_qa[6]})
                if args.version_2_with_negative:
                    inputs_student.update({"is_impossible": batch_qa[7]})
            outputs = model(**inputs_student, return_dict=False)

            # output the hidden states
            if args.output_hidden_states:
                loss, start_logits_stu, end_logits_stu, hidden_states_stu = outputs
            else:
                loss, start_logits_stu, end_logits_stu = outputs
                
            loss = loss * args.alpha_ce
                
            # KL loss if set
            if "kl" in args.losses or "klpw_plus_kl" in args.losses:
                if teacher is not None:
                    # Use the same batch for student and teacher
                    inputs_teacher = {
                        "input_ids": batch_qa[0],
                        "attention_mask": batch_qa[1],
                        "start_positions": batch_qa[3],
                        "end_positions": batch_qa[4],
                    }
                    if "token_type_ids" not in inputs_teacher:
                        inputs_teacher["token_type_ids"] = (
                            None if args.teacher_type == "xlm" else batch_qa[2]
                        )
                    with torch.no_grad():
                        outputs_tea = teacher(
                            input_ids=inputs_teacher["input_ids"],
                            token_type_ids=inputs_teacher["token_type_ids"],
                            attention_mask=inputs_teacher["attention_mask"],
                            return_dict=False,
                        )
                        if args.output_hidden_states:
                            (
                                start_logits_tea,
                                end_logits_tea,
                                hidden_states_tea,
                            ) = outputs_tea
                        else:
                            start_logits_tea, end_logits_tea = outputs_tea

                    assert start_logits_tea.size() == start_logits_stu.size()
                    assert end_logits_tea.size() == end_logits_stu.size()

                    loss_kl, alpha_kl = compute_distillation_loss(
                        args,
                        inputs_teacher["start_positions"],
                        inputs_teacher["end_positions"],
                        start_logits_stu,
                        start_logits_tea,
                        end_logits_stu,
                        end_logits_tea,
                        type=args.kl_type,
                    )

                    loss += alpha_kl * loss_kl

                if "klpw_plus_kl" in args.losses:
                    # apply parallel-wise knowledge distillation
                    if teacher is not None:
                        # Use batch with parallel qa examples
                        inputs_teacher = {
                            "input_ids": batch_parallel_qa[0],
                            "attention_mask": batch_parallel_qa[1],
                            "start_positions": batch_parallel_qa[3],
                            "end_positions": batch_parallel_qa[4],
                        }
                        if "token_type_ids" not in inputs_teacher:
                            inputs_teacher["token_type_ids"] = (
                                None
                                if args.teacher_type == "xlm"
                                else batch_parallel_qa[2]
                            )
                        with torch.no_grad():
                            outputs_tea = teacher(
                                input_ids=inputs_teacher["input_ids"],
                                token_type_ids=inputs_teacher["token_type_ids"],
                                attention_mask=inputs_teacher["attention_mask"],
                                return_dict=False,
                            )
                            if args.output_hidden_states:
                                (
                                    start_logits_tea,
                                    end_logits_tea,
                                    hidden_states_tea,
                                ) = outputs_tea
                            else:
                                start_logits_tea, end_logits_tea = outputs_tea

                        assert start_logits_tea.size() == start_logits_stu.size()
                        assert end_logits_tea.size() == end_logits_stu.size()

                        loss_klpw, alpha_klpw = compute_distillation_loss(
                            args,
                            inputs_teacher["start_positions"],
                            inputs_teacher["end_positions"],
                            start_logits_stu,
                            start_logits_tea,
                            end_logits_stu,
                            end_logits_tea,
                            type=args.kl_type,
                        )
                        loss += alpha_klpw * loss_klpw

            elif "klpw" in args.losses:
                # apply parallel-wise knowledge distillation
                if teacher is not None:
                    # Use batch with parallel qa examples
                    inputs_teacher = {
                        "input_ids": batch_parallel_qa[0],
                        "attention_mask": batch_parallel_qa[1],
                        "start_positions": batch_parallel_qa[3],
                        "end_positions": batch_parallel_qa[4],
                    }
                    if "token_type_ids" not in inputs_teacher:
                        inputs_teacher["token_type_ids"] = (
                            None
                            if args.teacher_type == "xlm"
                            else batch_parallel_qa[2]
                        )
                    with torch.no_grad():
                        outputs_tea = teacher(
                            input_ids=inputs_teacher["input_ids"],
                            token_type_ids=inputs_teacher["token_type_ids"],
                            attention_mask=inputs_teacher["attention_mask"],
                            return_dict=False,
                        )
                        if args.output_hidden_states:
                            (
                                start_logits_tea,
                                end_logits_tea,
                                hidden_states_tea,
                            ) = outputs_tea
                        else:
                            start_logits_tea, end_logits_tea = outputs_tea
                    assert start_logits_tea.size() == start_logits_stu.size()
                    assert end_logits_tea.size() == end_logits_stu.size()
                    loss_klpw, alpha_klpw = compute_distillation_loss(
                        args,
                        inputs_teacher["start_positions"],
                        inputs_teacher["end_positions"],
                        start_logits_stu,
                        start_logits_tea,
                        end_logits_stu,
                        end_logits_tea,
                        type=args.kl_type,
                    )
                    loss += alpha_klpw * loss_klpw

            if args.n_gpu > 1:
                loss = (
                    loss.mean()
                )  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                # If not provided, set default value for logging step to the number of steps in each epoch
                logging_steps = (
                    args.logging_steps if args.logging_steps else epoch_steps
                )
                if (
                    args.local_rank in [-1, 0]
                    and logging_steps > 0
                    and global_step % logging_steps == 0
                ):
                    try:
                        tb_writer.add_scalar("alpha_kl", alpha_kl, global_step)
                    except Exception:
                        pass
                    try:
                        tb_writer.add_scalar("alpha_klpw", alpha_klpw, global_step)
                    except Exception:
                        pass
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / logging_steps, global_step
                    )
                    logging_loss = tr_loss

                # Evaluate on dev set
                eval_steps = args.eval_steps if args.eval_steps else epoch_steps
                if (
                    args.local_rank == -1
                    and args.evaluate_during_training
                    and global_step % eval_steps == 0
                ):
                    # Only evaluate when single GPU otherwise metrics may not average well
                    logger.info(f"Evaluate on file: {args.predict_file}")
                    prefix = f"{os.path.basename(args.predict_file).replace('.json', '')}_checkpoint-{global_step}"
                    result = evaluate(args, model, tokenizer, prefix=f"{prefix}")
                    # result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
                    result.update({"testset": prefix})
                    logger.info("Results: {}".format(dict(result)))

                    # Compute aggregate G-XLT score
                    f1_mean = result["f1"]
                    tb_writer.add_scalar("eval_f1_mean", f1_mean, global_step)

                    # save results to file
                    results_file = f"{args.output_dir}/eval_results_{prefix}_checkpoint-{global_step}"
                    with open(results_file, "a") as fn:
                        fn.write(str(dict(result)) + "\n")

                    if f1_mean >= best_validation_score:
                        best_validation_score = f1_mean

                        # save the best model
                        # Take care of distributed/parallel training
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)

                        torch.save(
                            args, os.path.join(args.output_dir, "training_args.bin")
                        )
                        logger.info(
                            "Saving best validationmodel checkpoint to %s",
                            args.output_dir,
                        )

                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(args.output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(args.output_dir, "scheduler.pt"),
                        )
                        logger.info(
                            "Saving optimizer and scheduler states to %s",
                            args.output_dir,
                        )

                save_steps = args.save_steps if args.save_steps else epoch_steps
                if (
                    args.local_rank in [-1, 0]
                    and save_steps > 0
                    and global_step % save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_validation_score


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(
        args, tokenizer, evaluate=True, output_examples=True
    )

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if args.model_type != "distilbert":
                # XLM don't use segment_ids
                inputs["token_type_ids"] = (
                    None if args.model_type == "xlm" else batch[2]
                )
            example_indices = batch[3]
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})

            outputs = model(**inputs, return_dict=False)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                if args.output_hidden_states:
                    start_logits, end_logits, _ = output
                else:
                    start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info(
        "  Evaluation done in total %f secs (%f sec per example)",
        evalTime,
        evalTime / len(dataset),
    )

    # Compute predictions
    output_prediction_file = os.path.join(
        args.eval_dir if args.eval_dir else args.output_dir,
        "predictions_{}.json".format(prefix),
    )
    output_nbest_file = os.path.join(
        args.eval_dir if args.eval_dir else args.output_dir,
        "nbest_predictions_{}.json".format(prefix),
    )

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            args.output_dir, "null_odds_{}.json".format(prefix)
        )
    else:
        output_null_log_odds_file = None

    if args.model_type in ["xlnet", "xlm"]:
        # XLNet uses a more complex post-processing procedure
        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            model.config.start_n_top,
            model.config.end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    # Load data features from cache or dataset file
    cache_eval_dir = args.cache_eval_dir if args.cache_eval_dir else args.output_dir
    if evaluate:
        cached_features_file = os.path.join(
            cache_eval_dir,
            "cached_{}_{}".format(
                os.path.basename(args.predict_file), str(args.max_seq_length)
            ),
        )
    else:
        cached_features_file = os.path.join(
            args.output_dir, "cached_{}_{}".format("train", str(args.max_seq_length))
        )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)

        try:
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        except KeyError:
            raise DeprecationWarning(
                "You seem to be loading features from an older version of this script please delete the "
                "file %s in order for it to be created again" % cached_features_file
            )
    else:
        logger.info("Creating features from dataset file at %s", cache_eval_dir)
        if evaluate:
            processor = (
                SquadV2Processor()
                if args.version_2_with_negative
                else SquadV1Processor()
            )
            examples = processor.get_dev_examples(
                args.data_dir, filename=args.predict_file
            )

            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=not evaluate,
                return_dataset="pt",
                threads=args.threads,
            )
        else:
            if args.parallel_qa:
                processor = SquadProcessorMultilingual()
            else:
                processor = (
                    SquadV2Processor()
                    if args.version_2_with_negative
                    else SquadV1Processor()
                )

            examples = processor.get_train_examples(
                args.data_dir, filename=args.train_file
            )

            features, dataset = squad_convert_examples_to_features_parallel(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=not evaluate,
                return_dataset="pt",
                threads=args.threads,
            )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(
                {"features": features, "dataset": dataset, "examples": examples},
                cached_features_file,
            )

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--eval_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        default=None,
        type=str,
        help="The tensorboard logging dir.",
    )

    # Distillation parameters (optional)
    parser.add_argument(
        "--teacher_type",
        default=None,
        type=str,
        help="Teacher type."
        "Teacher tokenizer and student (model) tokenizer must output the same tokenization. Only for distillation.",
    )
    parser.add_argument(
        "--teacher_name_or_path",
        default=None,
        type=str,
        help="Path to the already SQuAD fine-tuned teacher model. Only for distillation.",
    )
    parser.add_argument(
        "--alpha_kl",
        default=1,
        type=float,
        help="Distillation Kullbacl-Leibler Loss linear weight. "
        "Only for distillation.",
    )
    parser.add_argument(
        "--alpha_ce",
        default=1,
        type=float,
        help="True SQuAD Cross-entropy loss linear weight. " "Only for distillation.",
    )
    parser.add_argument(
        "--temperature",
        default=2.0,
        type=float,
        help="Distillation temperature. Only for distillation.",
    )
    parser.add_argument(
        "--kl_type",
        default="fw",
        type=str,
        help="Type of KL divergence, forward, reverse or symmetric.",
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        help="Set the losses used during training, either standard QA Cross-Entropy loss, Kullback-Leibler loss for knowledge distillation or Mean-squared error loss between classification head's logits.",
    )
    parser.add_argument(
        "--use_map_loss_coefficients",
        action="store_true",
        help="Use MAP@k coefficients to balance KL and CE loss terms based on MAP of the top-k teacher predictions around the ground-thruth labels.",
    )
    parser.add_argument(
        "--use_mrr_loss_coefficients",
        action="store_true",
        help="Use MRR coefficients to balance KL and CE loss terms based on MAP of the first correct prediction in the top-k teacher predictions",
    )
    parser.add_argument(
        "--self_distil",
        action="store_true",
        help="Do not freeze the teacher at the beginning of the training and self-distill from the same model that is updated at each training step",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--cache_eval_dir",
        default=None,
        type=str,
        help="Cache directory for evaluations datasets",
    )
    parser.add_argument(
        "--parallel_qa",
        action="store_true",
        help="Process datasets with parallel QA examples.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_files",
        nargs="+",
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument("--prefixes", nargs="+", help="Prefixes for prediction files")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument("--constant_lr", action="store_true")
    parser.add_argument(
        "--linear_schedule_epoch",
        action="store_true",
        help="Linear decrease of learning rate only during the first epoch",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--train-layers",
        nargs="+",
        help="Select layers to train during training. The other ones will be freezed.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        help="Log every X updates steps. If not set, log at the end of each epoch ",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        help="Evaluate every X updates steps. If not set, log at the end of each epoch ",
    )
    parser.add_argument(
        "--save_steps", type=int, help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--save_best_model",
        action="store_true",
        help="Save best model based on F1 metric",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--save_best_model_at_end",
        action="store_true",
        help="Save best model based on the validation score.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--output_hidden_states",
        action="store_true",
        help="If set to True, change the default model configuration to output the model hidden states",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="Can be used for distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="Can be used for distant debugging."
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="multiple threads for converting example to features",
    )
    args = parser.parse_args()

    # write args to file
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as g:
        json.dump(args.__dict__, g, indent=2)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # set the output hidden states configuration to True for the student model
    if args.output_hidden_states:
        config.output_hidden_states = True
    # Fast tokenizer does work with the QA pipeline, as pointed out here: https://github.com/huggingface/transformers/issues/7735
    # Therefore, we load a slow tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.teacher_type is not None:
        assert args.teacher_name_or_path is not None
        assert args.alpha_kl + args.alpha_ce > 0.0
        assert (
            args.teacher_type != "distilbert"
        ), "We constraint teachers not to be of type DistilBERT."

        teacher_config_class, teacher_model_class, _ = MODEL_CLASSES[args.teacher_type]
        teacher_config = teacher_config_class.from_pretrained(
            args.teacher_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        # set the output hidden states configuration to True for the teacher model
        if args.output_hidden_states:
            teacher_config.output_hidden_states = True
        teacher = teacher_model_class.from_pretrained(
            args.teacher_name_or_path,
            config=teacher_config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        teacher.to(args.device)
    else:
        teacher = None

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, tokenizer, evaluate=False, output_examples=False
        )
        global_step, tr_loss, best_validation_score = train(
            args, train_dataset, model, tokenizer, teacher=teacher
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        if not args.save_best_model_at_end:
            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`

            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case
        )
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            # Reload the model
            # global_step = checkpoint.split("-")[-1] if checkpoints else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
            model.config.output_hidden_states = False
            model.to(args.device)

            # Evaluate
            for predict_file in args.predict_files:
                args.predict_file = predict_file
                prefix = f"{os.path.basename(args.predict_file).replace('.json', '')}"
                result = evaluate(args, model, tokenizer, prefix=prefix)

                result = {k: v for (k, v) in result.items()}
                results.update(result)

                logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
