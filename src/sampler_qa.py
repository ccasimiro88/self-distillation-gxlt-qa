import argparse
from distutils import debug
import random
import logging
import json
import os
import glob
import random
from ordered_set import OrderedSet
from collections import defaultdict
from statistics import mean, stdev
import math
from tqdm import tqdm
import pprint
from itertools import combinations


class SquadMultilingual:
    def __init__(self, squad_version, languages):
        self.squad_version = squad_version
        self.languages = languages

        self.mlqa_languages = ["en", "es", "de", "ar", "hi", "vi", "zh"]
        self.xquad_languages = ["en", "es", "de", "ar", "hi", "vi", "zh"] + [
            "ru",
            "th",
            "tr",
            "el",
        ]

    @staticmethod
    def _parse_file(file):
        with open(os.path.realpath(file)) as reader:
            dataset = json.load(reader)
        return dataset

    @staticmethod
    def qids(dataset):
        qids = []
        for data in dataset["data"]:
            for par in data["paragraphs"]:
                for qa in par["qas"]:
                    qids.append(qa["id"])
        return qids

    def load_dataset(self, files):
        dataset_joint = {"version": None, "data": []}
        datasets = []
        for file in files:
            dataset = self._parse_file(file)
            assert (
                str(dataset["version"])[0] == self.squad_version
            ), f"Using invalid SQUAD version {dataset['version']}"
            datasets.append(dataset)

        for dataset in datasets:
            dataset_joint["data"].extend(dataset["data"])
        random.shuffle(dataset_joint["data"])
        return dataset_joint

    def select_data_files(self, data_dirs, data_types, lang, split):
        files = []

        for data_dir, data_type in zip(data_dirs, data_types):
            if data_type == "mlqa":
                if lang in self.languages:
                    if split == "train":
                        files.extend(
                            glob.glob(
                                f"{data_dir}/*context-{lang}-question-{lang}.json"
                            )
                        )
                    elif split == "validation":
                        files.extend(glob.glob(f"{data_dir}/*context-{lang}*"))
            elif data_type == "xquad":
                if lang in self.xquad_languages:
                    files.append(glob.glob(f"{data_dir}/xquad.{lang}.json")[0])
        return files

    def create_qa(
        self,
        data_dirs,
        data_types,
        parallel_qa_type,
        num_tgt_langs,
        parallel_qa,
        lang_par_question,
        lang_par_context,
        non_en_centric,
        split,
    ):
        # Flatten out the content (qid, context, questions, answers and start indexes) and group by language
        qid_to_content_per_lang = defaultdict(lambda: defaultdict(dict))
        for lang in self.languages:
            files = self.select_data_files(data_dirs, data_types, lang, split)
            dataset = self.load_dataset(files)
            for data in dataset["data"]:
                title = data["title"]
                for par in data["paragraphs"]:
                    context = par["context"]

                    for qa in par["qas"]:
                        answers = qa["answers"]
                        question = qa["question"]
                        qid = qa["id"]
                        qid_to_content_per_lang[qid][lang] = {
                            "title": title,
                            "context": context,
                            "answers": answers,
                            "question": question,
                        }

        examples_count = defaultdict(int)
        pair_count = defaultdict(int)
        question_question_parallel_tea_pairs = defaultdict(int)
        dataset_joint = {"version": self.squad_version, "data": []}
        # Select a list of examples by question ids
        qids = list(OrderedSet(qid_to_content_per_lang.keys()))
        for qid in tqdm(qids, desc="Sampling from qid"):
            if non_en_centric:
                qid_tgt_languages = set(
                    lang for lang in qid_to_content_per_lang[qid].keys()
                ).intersection(self.languages)
            else:
                qid_tgt_languages = set(
                    lang for lang in qid_to_content_per_lang[qid].keys() if lang != "en"
                ).intersection(self.languages)

            if num_tgt_langs >= len(qid_tgt_languages) or num_tgt_langs == 0:
                langs_tgt_sampled = set(
                    lang for lang in qid_to_content_per_lang[qid].keys()
                ).intersection(self.languages)
            else:
                if non_en_centric:
                    langs_tgt_sampled = random.sample(
                        qid_tgt_languages, k=num_tgt_langs
                    )
                else:
                    langs_tgt_sampled = random.sample(
                        qid_tgt_languages, k=num_tgt_langs
                    ) + ["en"]

            # join all parallel pairs
            for lang_question in langs_tgt_sampled:
                for lang_context in langs_tgt_sampled:
                    if parallel_qa:
                        if parallel_qa_type == "q-random":
                            # distill from bilingual random question-parallel examples
                            lang_par_question = random.choice(
                                [
                                    lang
                                    for lang in langs_tgt_sampled
                                    if lang != lang_question
                                ]
                            )
                            lang_par_context = lang_context

                        elif parallel_qa_type == "q-en-diag":
                            # distill from bilinugal pairs with English question of from diagonal depending on the lang_question
                            if lang_question == "en":
                                lang_par_question = lang_par_context = lang_context

                            elif lang_question == lang_context:
                                lang_par_question = "en"
                                lang_par_context = lang_context
                        else:
                            # select specific question and context languages
                            lang_par_question = (
                                lang_par_question
                                if lang_par_question
                                else lang_question
                            )
                            lang_par_context = (
                                lang_par_context if lang_par_context else lang_context
                            )
                        dataset_joint["data"].append(
                            {
                                "title": qid_to_content_per_lang[qid][lang_question][
                                    "title"
                                ],
                                "paragraphs": [
                                    {
                                        "context": qid_to_content_per_lang[qid][
                                            lang_context
                                        ]["context"],
                                        "context_parallel": qid_to_content_per_lang[
                                            qid
                                        ][lang_par_context]["context"],
                                        "qas": [
                                            {
                                                "question": qid_to_content_per_lang[
                                                    qid
                                                ][lang_question]["question"],
                                                "question_parallel": qid_to_content_per_lang[
                                                    qid
                                                ][
                                                    lang_par_question
                                                ][
                                                    "question"
                                                ],
                                                "answers": qid_to_content_per_lang[qid][
                                                    lang_context
                                                ]["answers"],
                                                "answers_parallel": qid_to_content_per_lang[
                                                    qid
                                                ][
                                                    lang_par_context
                                                ][
                                                    "answers"
                                                ],
                                                "id": f"{qid}_{lang_question}-{lang_context}",
                                            }
                                        ],
                                    }
                                ],
                            }
                        )
                        examples_count[qid] += 1
                        pair_count[f"{lang_question}-{lang_context}"] += 1
                        question_question_parallel_tea_pairs[
                            f"{lang_question}-{lang_context},{lang_par_question}-{lang_par_context}"
                        ] += 1

                    if num_tgt_langs == 0:
                        if lang_question == lang_context:
                            dataset_joint["data"].append(
                                {
                                    "title": qid_to_content_per_lang[qid][
                                        lang_question
                                    ]["title"],
                                    "paragraphs": [
                                        {
                                            "context": qid_to_content_per_lang[qid][
                                                lang_context
                                            ]["context"],
                                            "context_parallel": qid_to_content_per_lang[
                                                qid
                                            ][lang_context]["context"],
                                            "qas": [
                                                {
                                                    "question": qid_to_content_per_lang[
                                                        qid
                                                    ][lang_question]["question"],
                                                    "question_parallel": qid_to_content_per_lang[
                                                        qid
                                                    ][
                                                        lang_context
                                                    ][
                                                        "question"
                                                    ],
                                                    "answers": qid_to_content_per_lang[
                                                        qid
                                                    ][lang_context]["answers"],
                                                    "answers_parallel": qid_to_content_per_lang[
                                                        qid
                                                    ][
                                                        lang_context
                                                    ][
                                                        "answers"
                                                    ],
                                                    "id": f"{qid}_{lang_question}-{lang_context}",
                                                }
                                            ],
                                        }
                                    ],
                                }
                            )
                            examples_count[qid] += 1
                            pair_count[f"{lang_question}-{lang_context}"] += 1
                    else:
                        dataset_joint["data"].append(
                            {
                                "title": qid_to_content_per_lang[qid][lang_question][
                                    "title"
                                ],
                                "paragraphs": [
                                    {
                                        "context": qid_to_content_per_lang[qid][
                                            lang_context
                                        ]["context"],
                                        "context_parallel": qid_to_content_per_lang[
                                            qid
                                        ][lang_context]["context"],
                                        "qas": [
                                            {
                                                "question": qid_to_content_per_lang[
                                                    qid
                                                ][lang_question]["question"],
                                                "question_parallel": qid_to_content_per_lang[
                                                    qid
                                                ][
                                                    lang_context
                                                ][
                                                    "question"
                                                ],
                                                "answers": qid_to_content_per_lang[qid][
                                                    lang_context
                                                ]["answers"],
                                                "answers_parallel": qid_to_content_per_lang[
                                                    qid
                                                ][
                                                    lang_context
                                                ][
                                                    "answers"
                                                ],
                                                "id": f"{qid}_{lang_question}-{lang_context}",
                                            }
                                        ],
                                    }
                                ],
                            }
                        )
                        examples_count[qid] += 1
                        pair_count[f"{lang_question}-{lang_context}"] += 1

        qas_count = sum(examples_count.values())
        qid_count_unique = len(examples_count.keys())
        langs_mean_frequency = round(mean([f for l, f in pair_count.items()]), 2)
        langs_variance_frequency = round(stdev([f for l, f in pair_count.items()]), 2)

        info_joint = {
            "number_qa_examples": qas_count,
            "qid_count_unique": qid_count_unique,
            "langs_average_frequency": langs_mean_frequency,
            "langs_variance_frequency": langs_variance_frequency,
            "pair_count": dict(pair_count),
            "question_question_parallel_pairs": dict(
                question_question_parallel_tea_pairs
            ),
        }
        return dataset_joint, info_joint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-dirs",
        type=str,
        nargs="+",
        required=True,
        help="directory with the Squad file format in different languages",
    )
    parser.add_argument("--train-data-types", type=str, nargs="+", required=True)
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        default=["en", "es", "de", "ar", "vi", "hi", "zh"],
        help="list of languages to use for the training set",
    )
    parser.add_argument("--squad_version", type=str, default="1")
    parser.add_argument(
        "--valid-data-dirs",
        nargs="+",
        help="directory with the Squad file format in different languages",
    )
    parser.add_argument("--valid-data-types", nargs="+", default="mlqa")
    parser.add_argument(
        "--parallel-qa",
        action="store_true",
        default=False,
        help="Create parallel QA examples for knowledge distillation",
    )
    parser.add_argument(
        "--lang-par-question",
        type=str,
        default=None,
        help="Select the language of the question to distill from suring class-wise distillation",
    )
    parser.add_argument(
        "--lang-par-context",
        type=str,
        default=None,
        help="Select the language of the question to distill from suring class-wise distillation",
    )
    parser.add_argument("--create-valid-split", action="store_true")
    parser.add_argument("--non-en-centric", action="store_true", default=False)
    parser.add_argument("--fixed-par-langs", nargs="+")
    parser.add_argument(
        "--simple-join",
        type=str,
        choices=["mono", "mixed"],
        help="join all data examples",
    )
    parser.add_argument(
        "--num-tgt-langs",
        type=int,
        help="Number of target languages for bilingual pairs",
    )
    parser.add_argument(
        "--parallel-qa-type",
        type=str,
        choices=["q-random", "q-en-diag"],
        help="Type of qa pair",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="squad_sampler",
        help="directory where the generated file is written",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        filename=os.path.join(args.output_dir, "log.txt"),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    random.seed(args.seed)

    pp = pprint.PrettyPrinter(indent=4)

    logging.info(f"Generate cross-lingual SQUAD data...")
    logging.info(f"Merging cross-lingual pairs for languages: {args.languages}")

    # check args
    if args.parallel_qa:
        logging.info(f"Use parallel questions format for class-wise distillation")
    split = "train"
    data_generator = SquadMultilingual(
        squad_version=args.squad_version, languages=args.languages
    )

    dataset, info = data_generator.create_qa(
        data_dirs=args.train_data_dirs,
        data_types=args.train_data_types,
        parallel_qa_type=args.parallel_qa_type,
        num_tgt_langs=args.num_tgt_langs,
        parallel_qa=args.parallel_qa,
        lang_par_question=args.lang_par_question,
        lang_par_context=args.lang_par_context,
        non_en_centric=args.non_en_centric,
        split=split,
    )
    info_tot = {split: info}
    logging.info(
        f"Train set: Created {info['number_qa_examples']} question-answer pairs"
    )
    logging.info(
        f"Train set: Collected questions with average and variance frequency of"
        f" ({info['langs_average_frequency']}, {info['langs_variance_frequency']})"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    random.shuffle(dataset["data"])

    file = os.path.join(args.output_dir, "train.json")
    with open(file, "w") as ft:
        json.dump(dataset, ft)
    logging.info(
        f"Write train to file:" f"\n{file} " f"({len(data_generator.qids(dataset))})"
    )

    if args.valid_data_dirs and args.valid_data_types:
        # Create validation set considering all the combinations
        split = "validation"

        dataset, info = data_generator.create_qa(
            data_dirs=args.valid_data_dirs,
            data_types=args.valid_data_types,
            parallel_qa_type="mixed",
            num_tgt_langs=len(args.languages),
            parallel_qa=False,
            lang_par_question=args.lang_par_question,
            lang_par_context=args.lang_par_context,
            non_en_centric=args.non_en_centric,
            split=split,
        )
        info_tot.update({split: info})
        logging.info(
            f"Validation set: Created {info['number_qa_examples']} question-answer pairs"
        )
        logging.info(
            f"Validation set: Collected questions with average and variance frequency of"
            f" ({info['langs_average_frequency']}, {info['langs_variance_frequency']})"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        file = os.path.join(args.output_dir, "mlqa-dev-gxlt.json")
        with open(file, "w") as ft:
            json.dump(dataset, ft)
        logging.info(
            f"Write Validation to file:"
            f"\n{file} "
            f"({len(data_generator.qids(dataset))})"
        )
        pp.pprint(info_tot)

    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f, open(
        os.path.join(args.output_dir, "data_args.json"), "w"
    ) as g:
        json.dump(info_tot, f, indent=2)
        json.dump(args.__dict__, g, indent=2)
