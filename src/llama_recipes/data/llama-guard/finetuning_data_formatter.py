import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence


@dataclass
class Category:
    name: str
    description: str


@dataclass
class Guidelines:
    categories: Sequence[Category]
    category_code_prefix: str = "C"


class ExplanationPosition(Enum):
    BEFORE_DECISION = 0
    AFTER_DECISION = 1


@dataclass
class LlamaGuardPromptConfig:
    instructions_format_string: str
    should_include_category_descriptions: bool
    should_shuffle_category_codes: bool = True


@dataclass
class LlamaGuardGenerationConfig:
    should_include_violation_codes: bool
    explanation_position: Optional[ExplanationPosition]


@dataclass
class AugmentationConfigs:
    probability_to_augment_with_safe_examples_with_empty_responses: float = 0
    should_augment_with_examples_with_dropped_nonviolated_prompt_categories: bool = True
    should_augment_with_examples_with_dropped_violated_and_nonviolated_prompt_categories: bool = (
        False
    )


@dataclass
class FormatterConfigs:
    guidelines: Guidelines
    llama_guard_prompt_config: LlamaGuardPromptConfig
    llama_guard_generation_config: LlamaGuardGenerationConfig
    augmentation_configs: AugmentationConfigs
    # Allows subsequent reruns of the data formatter to reuse a stable seed for reproducibility
    random_seed: int = 42


@dataclass
class TrainingExample:
    prompt: str
    response: str
    violated_category_codes: list[str]
    label: Literal["safe", "unsafe"]
    explanation: str


def create_formatted_finetuning_examples(
    training_examples: Sequence[TrainingExample],
    formatter_configs: FormatterConfigs,
) -> list[str]:
    random.seed(formatter_configs.random_seed)

    indices_of_all_categories = range(len(formatter_configs.guidelines.categories))

    to_return = []

    for training_example in training_examples:
        to_return.append(
            _create_formatted_finetuning_example(
                training_example,
                formatter_configs,
                category_indeces_to_include_in_llama_guard_prompt=list(
                    indices_of_all_categories
                ),
            )
        )

        _maybe_augment_with_safe_example_with_empty_response(
            training_example, to_return, indices_of_all_categories, formatter_configs
        )

        _maybe_augment_with_examples_with_dropped_prompt_categories(
            training_example, to_return, indices_of_all_categories, formatter_configs
        )

    print(f"Successfully created {len(to_return)} formatted finetuning examples.")

    return to_return


def _create_formatted_finetuning_example(
    training_example: TrainingExample,
    formatter_configs: FormatterConfigs,
    category_indeces_to_include_in_llama_guard_prompt: List[int],
):
    if formatter_configs.should_shuffle_category_codes:
        random.shuffle(category_indeces_to_include_in_llama_guard_prompt)
    else:
        category_indeces_to_include_in_llama_guard_prompt = sorted(
            category_indeces_to_include_in_llama_guard_prompt
        )

    conversation = {"human": training_example.prompt}

    if not _is_prompt_only_example(training_example):
        conversation["chatbot"] = training_example.response

    llama_guard_prompt = _create_llama_guard_prompt(
        conversation,
        category_indeces_to_include_in_llama_guard_prompt,
        formatter_configs,
    )

    llama_guard_generation = _create_llama_guard_generation(
        training_example,
        formatter_configs,
        category_indeces_to_include_in_llama_guard_prompt,
    )

    return f"{llama_guard_prompt} {llama_guard_generation}"


def _is_prompt_only_example(training_example: TrainingExample) -> bool:
    return training_example.response == "N/A"


def _create_llama_guard_prompt(
    conversation: Dict[str, str],
    category_indices_to_include: List[int],
    formatter_configs: FormatterConfigs,
) -> str:
    full_guidelines_text = ""

    for (
        rewritten_category_index_for_current_prompt,
        original_category_index,
    ) in enumerate(category_indices_to_include):
        category = formatter_configs.guidelines.categories[original_category_index]

        full_guidelines_text += f"\n{formatter_configs.guidelines.category_code_prefix}{rewritten_category_index_for_current_prompt + 1}: {category.name}. "

        if (
            formatter_configs.llama_guard_prompt_config.should_include_category_descriptions
        ):
            full_guidelines_text += f"\n{category.description}"

    return formatter_configs.llama_guard_prompt_config.instructions_format_string.format_map(
        {
            "guidelines": full_guidelines_text,
            "conversation": _serialize_conversation(conversation),
        }
    )


def _serialize_conversation(conversation: Dict[str, str]) -> str:
    conversation = []

    for speaker, message in conversation.items():
        conversation.append(f"{speaker}: {message}")

    return "\n\n".join(conversation)


def _create_llama_guard_generation(
    training_example: TrainingExample,
    formatter_configs: FormatterConfigs,
    category_indices_to_include_in_llama_guard_prompt: List[int],
) -> str:
    to_return = training_example.label

    if training_example.label == "unsafe" and formatter_configs.llama_guard_generat:
        original_violated_category_indices = set(
            _convert_category_codes_to_indices(
                training_example.violated_category_codes,
                formatter_configs.guidelines.category_code_prefix,
            )
        )

        map_of_original_category_indices_to_rewritten_category_codes = (
            _get_map_of_original_category_indices_to_rewritten_category_codes(
                formatter_configs, category_indices_to_include_in_llama_guard_prompt
            )
        )

        rewritten_violated_category_codes = [
            map_of_original_category_indices_to_rewritten_category_codes[
                original_violated_index
            ]
            for original_violated_index in original_violated_category_indices
        ]

        to_return += "\n"
        to_return += ",".join(rewritten_violated_category_codes)

    explanation_position = (
        formatter_configs.llama_guard_generation_config.explanation_position
    )

    if explanation_position == ExplanationPosition.BEFORE_DECISION:
        to_return = f"Explanation: {training_example.explanation}\n{to_return}"
    elif explanation_position == ExplanationPosition.AFTER_DECISION:
        to_return = f"{to_return}\nExplanation: {training_example[formatter_configs.explanation_key]}"

    return to_return


def _get_map_of_original_category_indices_to_rewritten_category_codes(
    formatter_configs: FormatterConfigs,
    category_indices_to_include_in_llama_guard_prompt: List[int],
):
    return {
        original_category_index: formatter_configs.guidelines.category_code_prefix
        + str(rewritten_category_index + 1)
        for rewritten_category_index, original_category_index in enumerate(
            category_indices_to_include_in_llama_guard_prompt
        )
    }


def _maybe_augment_with_safe_example_with_empty_response(
    training_example: TrainingExample,
    formatted_examples_to_build: list[dict[str, str]],
    indices_of_all_categories: range,
    formatter_configs: FormatterConfigs,
) -> None:
    if (
        not _is_prompt_only_example(training_example)
        and random.random()
        < formatter_configs.augmentation_configs.probability_to_augment_with_safe_examples_with_empty_responses
    ):
        training_example_copy = copy.deepcopy(training_example)
        training_example_copy.response = ""
        training_example_copy.label = "safe"
        training_example_copy.violated_category_codes = []

        formatted_examples_to_build.append(
            _create_formatted_finetuning_example(
                training_example_copy,
                formatter_configs,
                category_indeces_to_include_in_llama_guard_prompt=list(
                    indices_of_all_categories
                ),
            )
        )


def _maybe_augment_with_examples_with_dropped_prompt_categories(
    training_example: TrainingExample,
    formatted_examples_to_build: list[dict[str, str]],
    indices_of_all_categories: range,
    formatter_configs: FormatterConfigs,
) -> None:
    violated_category_indices = _convert_category_codes_to_indices(
        training_example.violated_category_codes,
        formatter_configs.guidelines.category_code_prefix,
    )

    nonviolated_category_indices = list(
        set(indices_of_all_categories) - set(violated_category_indices)
    )

    _maybe_augment_with_example_with_dropped_nonviolated_prompt_categories(
        training_example,
        formatted_examples_to_build,
        indices_of_all_categories,
        nonviolated_category_indices,
        formatter_configs,
    )

    _maybe_augment_with_example_with_dropped_violated_and_nonviolated_prompt_categories(
        training_example,
        formatted_examples_to_build,
        indices_of_all_categories,
        violated_category_indices,
        nonviolated_category_indices,
        formatter_configs,
    )


def _convert_category_codes_to_indices(codes: list[str], code_prefix: str) -> list[int]:
    # The string indexing of GLs is 1-based, so we convert to 0-based here for consistency with python indexing
    return [int(code.lstrip(code_prefix)) - 1 for code in codes]


def _maybe_augment_with_example_with_dropped_nonviolated_prompt_categories(
    training_example: TrainingExample,
    formatted_examples_to_build: list[dict[str, str]],
    indices_of_all_categories: range,
    nonviolated_category_indices: list[int],
    formatter_configs: FormatterConfigs,
) -> None:
    if (
        not formatter_configs.augmentation_configs.should_augment_with_examples_with_dropped_nonviolated_prompt_categories
    ):
        pass

    number_of_categories_to_drop = random.randint(0, len(nonviolated_category_indices))

    if number_of_categories_to_drop == len(indices_of_all_categories):
        number_of_categories_to_drop -= 1

    dropped_category_indices = random.sample(
        nonviolated_category_indices, number_of_categories_to_drop
    )

    retained_category_indices = list(
        set(indices_of_all_categories) - (set(dropped_category_indices))
    )

    formatted_examples_to_build.append(
        _create_formatted_finetuning_example(
            training_example,
            formatter_configs,
            category_indeces_to_include_in_llama_guard_prompt=retained_category_indices,
        )
    )


def _maybe_augment_with_example_with_dropped_violated_and_nonviolated_prompt_categories(
    training_example: TrainingExample,
    formatted_examples_to_build: list[dict[str, str]],
    indices_of_all_categories: range,
    violated_category_indices: list[int],
    nonviolated_category_indices: list[int],
    formatter_configs: FormatterConfigs,
) -> None:
    if (
        training_example.label == "safe"
        or not formatter_configs.augmentation_configs.should_augment_with_examples_with_dropped_violated_and_nonviolated_prompt_categories
    ):
        pass

    random_nonviolated_category_indices_to_drop = random.sample(
        nonviolated_category_indices,
        random.randint(0, len(nonviolated_category_indices) - 1),
    )

    set_of_retained_category_indices = (
        set(indices_of_all_categories)
        - set(violated_category_indices)
        - set(random_nonviolated_category_indices_to_drop)
    )

    training_example_copy = copy.deepcopy(training_example)
    training_example_copy.label = "safe"
    training_example_copy.violated_category_codes = []

    formatted_examples_to_build.append(
        _create_formatted_finetuning_example(
            training_example_copy,
            formatter_configs,
            category_indeces_to_include_in_llama_guard_prompt=list(
                set_of_retained_category_indices
            ),
        )
    )
