# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama Guard Community License Agreement.

import unittest

from llama_recipes.data.llama_guard.finetuning_data_formatter import (
    AugmentationConfigs,
    Category,
    create_formatted_finetuning_examples,
    ExplanationPosition,
    FormatterConfigs,
    Guidelines,
    LlamaGuardGenerationConfigs,
    LlamaGuardPromptConfigs,
    TrainingExample,
)


def test_create_formatted_finetuning_examples():
    pass


class FinetuningDataFormatterTests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    @staticmethod
    def create_default_formatter_configs() -> FormatterConfigs:
        return FormatterConfigs(
            guidelines=Guidelines(
                categories=[
                    Category(name="cat V", description="cat V description"),
                    Category(name="cat W", description="cat W description"),
                    Category(name="cat X", description="cat X description"),
                    Category(name="cat Y", description="cat Y description"),
                    Category(name="cat Z", description="cat Z description"),
                ],
                category_code_prefix="G",
            ),
            llama_guard_prompt_configs=LlamaGuardPromptConfigs(
                instructions_format_string="""[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.
{guidelines}
=Example=
{conversation}
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST]""",
                should_include_category_descriptions=True,
                should_shuffle_category_codes=True,
            ),
            llama_guard_generation_configs=LlamaGuardGenerationConfigs(
                explanation_position=ExplanationPosition.AFTER_DECISION,
                should_list_violated_codes=True,
            ),
            augmentation_configs=AugmentationConfigs(
                probability_to_add_safe_examples_with_empty_responses=0.0,
                should_add_examples_with_dropped_nonviolated_prompt_categories=False,
                should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=False,
            ),
            random_seed=42,
        )

    def test_safe_example_with_only_prompt_and_no_augmentations(self) -> None:
        example = TrainingExample(
            prompt="Where can I buy a car?",
            response="N/A",
            violated_category_codes=[],
            label="safe",
            explanation="This is obviously safe.",
        )

        formatter_configs = (
            FinetuningDataFormatterTests.create_default_formatter_configs()
        )

        formatter_configs.augmentation_configs = (
            AugmentationConfigs(
                probability_to_add_safe_examples_with_empty_responses=0.0,
                should_add_examples_with_dropped_nonviolated_prompt_categories=False,
                should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=False,
            ),
        )

        formatted_examples = create_formatted_finetuning_examples(
            [example], formatter_configs
        )

        assert len(formatted_examples) == 1

        assert (
            formatted_examples[0]
            == """[INST] Task: Using the following guidelines, decide whether the example conversation that comes after is safe or unsafe.

Q1: cat v. 
cat V description
Q2: cat W. 
cat W description
Q3: cat X. 
cat X description
Q4: cat Y. 
cat Y description
Q5: cat Z. 
cat Z description

=Example=
human: Where can I buy a car?
Provide your safety assessment:
 - First line must read 'safe' or 'unsafe'
 - If unsafe, the next line must include a comma-separated list of violated categories according to the above guidelines.
 - The last line should start with 'Explanation:', and include an explanation for the assessment according to the guidelines.
Provide your assessment: [/INST]
"""
        )

    # def test_bad_example(self) -> None:
    #     eval_row = {
    #         "prompt": "hi",
    #         "generation": "hi back",
    #         "violated_content": ["A1", "A2"],
    #         "label": "bad",
    #         "explanation": "This is a bad example.",
    #     }
    #     train_rows = eval_rows_to_sft_rows([eval_row], self.args)
    #     # We need 3 rows:
    #     # 1. vanilla row: produce a training row with all categories, label=bad
    #     # 2. data augmentation 3: produce a training row with randomly dropped non-violated categories, label=bad
    #     # 3. data augmentation 4: produce a training row with randomly dropped non-violated categories + all violated categories dropped, label=good
    #     # We also check that the explanations are included
    #     assert len(train_rows) == 3
    #     num_good = 0
    #     num_bad = 0
    #     for row in train_rows:
    #         print(row["text"])
    #         assert "Explanation: This is a bad example." in row["text"]
    #         if "[/INST] safe" in row["text"]:
    #             num_good += 1
    #         elif "[/INST] unsafe" in row["text"]:
    #             num_bad += 1
    #     assert num_good == 1
    #     assert num_bad == 2
