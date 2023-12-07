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
    category: Sequence[Category]
    category_code_prefix: str = "C"


class DecisionExplanationPosition(Enum):
    BEFORE_DECISION = 0
    AFTER_DECISION = 1


@dataclass
class LlamaGuardPromptTemplate:
    instructions: str
    should_include_category_descriptions: bool
    should_shuffle_category_codes_in_each_training_example: bool = True


@dataclass
class DatagenArgs:
    guidelines: Guidelines
    llama_guard_prompt_template: LlamaGuardPromptTemplate
    probability_to_augment_with_good_rows_with_empty_generation: float = 0
    should_augment_with_rows_with_dropped_nonviolated_prompt_categories: bool = True
    should_augment_with_rows_with_dropped_violated_and_nonviolated_prompt_categories: bool = (
        False
    )
    coarse_label_instruction_prob: float = 0.0
    random_seed: int = 42
    should_generations_include_violation_codes: bool
    # If present, specifies whether the explanation is shown before or after the decicion in the llama guard generation.
    explanation_position_in_llama_guard_generation: Optional[DecisionExplanationPosition]


@dataclass
class TrainingExample:
    prompt: str
    generation: str
    violated_categorie_codes: list[str]
    label: Literal["safe", "unsafe"]
    explanation: str


# TODO @nocommit HANDLE N/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA


def create_formatted_finetuning_examples(
    training_examples: Sequence[TrainingExample],
    args: DatagenArgs,
) -> list[str]:
    random.seed(args.random_seed)

    indices_of_all_categories = range(len(args.guidelines.categories))

    to_return = []

    for training_example in training_examples:
        to_return.append(
            create_formatted_finetuning_example(
                training_example,
                args,
                category_indeces_to_include_in_llama_guard_prompt=list(
                    indices_of_all_categories
                ),
            )
        )

        maybe_augment_with_good_row_with_empty_generation(
            training_example, to_return, indices_of_all_categories, args
        )

        maybe_augment_with_rows_with_dropped_prompt_categories(
            training_example, to_return, indices_of_all_categories, args
        )

    print(f"Successfully created {len(to_return)} formatted finetuning examples.")

    return to_return


def create_formatted_finetuning_example(
    training_example: TrainingExample,
    args: DatagenArgs,
    category_indeces_to_include_in_llama_guard_prompt: List[int],
):
    if args.should_shuffle_category_codes_in_each_training_example:
        random.shuffle(category_indeces_to_include_in_llama_guard_prompt)
    else:
        category_indeces_to_include_in_llama_guard_prompt = sorted(
            category_indeces_to_include_in_llama_guard_prompt
        )

    conversation = {"human": training_example.prompt}

    if training_example.generation != "N/A":
        conversation["chatbot"] = training_example.generation

    llama_guard_prompt = create_llama_guard_prompt(
        conversation,
        category_indeces_to_include_in_llama_guard_prompt,
        args,
    )

    llama_guard_generation = create_llama_guard_generation(
        training_example, args, category_indeces_to_include_in_llama_guard_prompt
    )

    return f"{llama_guard_prompt} {llama_guard_generation}"


def create_llama_guard_prompt(
    conversation: Dict[str, str],
    category_indices_to_include: List[int],
    args: DatagenArgs,
) -> str:
    full_guidelines_text = ""

    for (
        rewritten_category_index_for_current_prompt,
        original_category_index,
    ) in enumerate(category_indices_to_include):
        category = args.guidelines.categories[original_category_index]

        full_guidelines_text += f"\n{args.guidelines.category_code_prefix}{rewritten_category_index_for_current_prompt + 1}: {category.name}. "

        if args.llama_guard_prompt_template.should_include_category_descriptions:
            full_guidelines_text += f"\n{category.description}"

    return args.llama_guard_prompt_template.instructions.format_map(
        {
            "guidelines": full_guidelines_text,
            "conversation": serialize_conversation(conversation),
        }
    )


def serialize_conversation(conversation: Dict[str, str]) -> str:
    conversation = []

    for speaker, message in conversation.items():
        conversation.append(f"{speaker}: {message}")

    return "\n\n".join(conversation)


def create_llama_guard_generation(
    training_example: TrainingExample,
    args: DatagenArgs,
    category_indices_to_include_in_llama_guard_prompt: List[int],
) -> str:
    to_return = training_example.label

    if training_example.label == "unsafe" and args.prompt_template.requires_fine_labels:
        original_violated_category_indices = set(
            convert_category_codes_to_indices(
                training_example.violated_category_codes, args.guidelines.category_code_prefix
            )
        )

        map_of_original_category_indices_to_rewritten_category_codes = get_map_of_original_category_indices_to_rewritten_category_codes(args, category_indices_to_include_in_llama_guard_prompt)

        rewritten_violated_category_codes = [map_of_original_category_indices_to_rewritten_category_codes[original_violated_index] for original_violated_index in original_violated_category_indices]

        to_return += "\n"
        to_return += ','.join(rewritten_violated_category_codes)

    if args.explanation_position_in_llama_guard_generation is not None:
        if args.explanation_position_in_llama_guard_generation == DecisionExplanationPosition.BEFORE_DECISION:
            to_return = f"Explanation: {training_example.explanation}\n{to_return}"
        elif args.explanation_position_in_llama_guard_generation == DecisionExplanationPosition.AFTER_DECISION:
            to_return = f"{to_return}\nExplanation: {training_example[args.explanation_key]}"

    return to_return

def get_map_of_original_category_indices_to_rewritten_category_codes(
    args: DatagenArgs,
    category_indices_to_include_in_llama_guard_prompt: List[int]
):
    return {
        original_category_index: args.guidelines.category_code_prefix + str(rewritten_category_index + 1)
        for rewritten_category_index, original_category_index in enumerate(
            category_indices_to_include_in_llama_guard_prompt
        )
    }

def maybe_augment_with_good_row_with_empty_generation(
    training_example: TrainingExample,
    sft_rows_to_build: list[dict[str, str]],
    indices_of_all_categories: range,
    args: DatagenArgs,
) -> None:
    if (
        random.random()
        < args.probability_to_augment_with_good_rows_with_empty_generation
    ):
        training_example_copy = copy.deepcopy(training_example)
        training_example_copy.generation = ""
        training_example_copy.label = "safe"
        training_example_copy.violated_category_codes = []

        sft_rows_to_build.append(
            create_formatted_finetuning_example(
                training_example_copy,
                args,
                category_indeces_to_include_in_llama_guard_prompt=list(
                    indices_of_all_categories
                ),
            )
        )


def maybe_augment_with_rows_with_dropped_prompt_categories(
    training_example: TrainingExample,
    sft_rows_to_build: list[dict[str, str]],
    indices_of_all_categories: range,
    args: DatagenArgs,
) -> None:
    violated_category_indices = convert_category_codes_to_indices(
        training_example.violated_category_codes, args.guidelines.category_code_prefix
    )

    nonviolated_category_indices = list(
        set(indices_of_all_categories) - set(violated_category_indices)
    )

    maybe_augment_with_row_with_dropped_nonviolated_prompt_categories(
        training_example,
        sft_rows_to_build,
        indices_of_all_categories,
        nonviolated_category_indices,
        args,
    )

    maybe_augment_with_row_with_dropped_violated_and_nonviolated_prompt_categories(
        training_example,
        sft_rows_to_build,
        indices_of_all_categories,
        violated_category_indices,
        nonviolated_category_indices,
        args,
    )


def convert_category_codes_to_indices(codes: list[str], code_prefix: str) -> list[int]:
    # The string indexing of GLs is 1-based, so we convert to 0-based here for consistency with python indexing
    return [int(code.lstrip(code_prefix)) - 1 for code in codes]


def maybe_augment_with_row_with_dropped_nonviolated_prompt_categories(
    training_example: TrainingExample,
    sft_rows_to_build: list[dict[str, str]],
    indices_of_all_categories: range,
    nonviolated_category_indices: list[int],
    args: DatagenArgs,
) -> None:
    if not args.should_augment_with_rows_with_dropped_nonviolated_prompt_categories:
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

    sft_rows_to_build.append(
        create_formatted_finetuning_example(
            training_example,
            args,
            category_indeces_to_include_in_llama_guard_prompt=retained_category_indices,
        )
    )


def maybe_augment_with_row_with_dropped_violated_and_nonviolated_prompt_categories(
    training_example: TrainingExample,
    sft_rows_to_build: list[dict[str, str]],
    indices_of_all_categories: range,
    violated_category_indices: list[int],
    nonviolated_category_indices: list[int],
    args: DatagenArgs,
) -> None:
    if (
        training_example.label == "safe"
        or not args.should_augment_with_rows_with_dropped_violated_and_nonviolated_prompt_categories
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

    sft_rows_to_build.append(
        create_formatted_finetuning_example(
            training_example_copy,
            args,
            category_indeces_to_include_in_llama_guard_prompt=list(
                set_of_retained_category_indices
            ),
        )
    )
