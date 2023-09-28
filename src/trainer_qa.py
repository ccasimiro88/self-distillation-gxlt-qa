# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""

from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_distillation_loss

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions
            )
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(
        self,
        predict_dataset,
        predict_examples,
        ignore_keys=None,
        metric_key_prefix: str = "test",
    ):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(
            predict_examples, predict_dataset, output.predictions, "predict"
        )
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(
            predictions=predictions.predictions,
            label_ids=predictions.label_ids,
            metrics=metrics,
        )


class QuestionAnsweringTrainerDistillation(QuestionAnsweringTrainer):
    # TODO: change the name of alphas?
    def __init__(
        self, *args, teacher, temperature, alpha_qa, alpha_kl, train_method, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.temperature = temperature
        self.alpha_qa = alpha_qa
        self.alpha_kl = alpha_kl
        self.train_method = train_method

    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute teacher loss with Kullback-Leibler distance
        with torch.no_grad():
            self.teacher.eval()
            outputs_tea = self.teacher(
                input_ids=inputs["input_ids"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            start_logits_tea = outputs_tea["start_logits"]
            end_logits_tea = outputs_tea["end_logits"]

        # Compute student loss with Cross-entropy
        outputs_stu = model(**inputs, return_dict=True)
        loss_ce = outputs_stu["loss"]
        start_logits_stu = outputs_stu["start_logits"]
        end_logits_stu = outputs_stu["end_logits"]

        assert start_logits_tea.size() == start_logits_stu.size()
        assert end_logits_tea.size() == end_logits_stu.size()

        loss_kl = compute_distillation_loss(
            start_logits_stu,
            start_logits_tea,
            end_logits_stu,
            end_logits_tea,
            self.temperature,
            self.train_method,
        )

        # Compute the total loss as the sum of cross-entropy and kullback-leibler terms
        loss = self.alpha_kl * loss_kl + self.alpha_qa * loss_ce

        return (loss, outputs_stu, outputs_tea) if return_outputs else loss
