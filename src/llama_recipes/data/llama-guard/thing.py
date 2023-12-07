import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence


@dataclass
class Guideline:
    name: str
    description: str
    idx: int


@dataclass
class Guidelines:
    name: str
    guidelines: Sequence[Guideline]


# Explanations in the assessments can go before the decision ("PRE"), or after the decision ("POST")
class ExplanationPosition(Enum):
    PRE = 0
    POST = 1


@dataclass
class PromptTemplate:
    name: str
    instructions: str
    # Flag that indicates whether a prompt template uses the descriptions from the GLs
    should_include_guideline_descriptions: bool
    should_shuffle_guideline_codes_in_each_eval_row: bool = True
    # Flag that indicates whether a prompt template requires fine labels (GL categories) to be printed in assessment
    requires_fine_labels: bool


@dataclass
class DatagenArgs:
    guidelines: Guidelines
    llama_guard_prompt_template: PromptTemplate
    eval_type: str = "safety_response"
    guideline_code_prefix: str = "C"
    probability_to_augment_with_good_rows_with_empty_generation: float = 0
    should_augment_with_rows_with_dropped_nonviolated_prompt_guidelines: bool = True
    should_augment_with_rows_with_dropped_violated_and_nonviolated_prompt_guidelines: bool = (
        False
    )
    coarse_label_instruction_prob: float = 0.0
    random_seed: int = 42
    # If present, specifies whether the explanation is shown before or after the decicion in the llama guard generation.
    explanation_position_in_llama_guard_generation: Optional[ExplanationPosition]


@dataclass
class EvalRow:
    prompt: str
    generation: str
    violated_guideline_codes: list[str]
    label: Literal["safe", "bad"]
    explanation: str


# TODO @nocommit HANDLE N/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA


def eval_rows_to_sft_rows(
    eval_rows: Sequence[EvalRow],
    args: DatagenArgs,
) -> list[dict[str, str]]:
    random.seed(args.random_seed)

    indices_of_all_guidelines = range(len(args.guidelines.guidelines))

    to_return = []

    for eval_row in eval_rows:
        to_return.append(
            make_sft_row(
                eval_row,
                args,
                guideline_indeces_to_include_in_llama_guard_prompt=list(
                    indices_of_all_guidelines
                ),
            )
        )

        maybe_augment_with_good_row_with_empty_generation(
            eval_row, to_return, indices_of_all_guidelines, args
        )

        maybe_augment_with_rows_with_dropped_prompt_guidelines(
            eval_row, to_return, indices_of_all_guidelines, args
        )

    num_empty = to_return.count(None)

    to_return = [row for row in to_return if row is not None]

    print(f"Successfully created {len(to_return)} rows (discarded: {num_empty}).")

    return to_return


def make_sft_row(
    eval_row: EvalRow,
    args: DatagenArgs,
    guideline_indeces_to_include_in_llama_guard_prompt: List[int],
):
    if args.should_shuffle_guideline_codes_in_each_eval_row:
        random.shuffle(guideline_indeces_to_include_in_llama_guard_prompt)
    else:
        guideline_indeces_to_include_in_llama_guard_prompt = sorted(
            guideline_indeces_to_include_in_llama_guard_prompt
        )

    conversation = {"human": eval_row.prompt}

    if eval_row.generation != "N/A":
        conversation["chatbot"] = eval_row.generation

    llama_guard_prompt = create_llama_guard_prompt(
        conversation,
        guideline_indeces_to_include_in_llama_guard_prompt,
        args,
    )

    llama_guard_generation = create_llama_guard_generation(
        eval_row, args, guideline_indeces_to_include_in_llama_guard_prompt
    )

    return f"{llama_guard_prompt} {llama_guard_generation}"


def create_llama_guard_prompt(
    conversation: Dict[str, str],
    guideline_indices_to_include: List[int],
    args: DatagenArgs,
) -> str:
    full_guidelines_text = ""

    for (
        rewritten_guideline_index_for_current_prompt,
        original_guideline_index,
    ) in enumerate(guideline_indices_to_include):
        guideline = args.guidelines.guidelines[original_guideline_index]

        full_guidelines_text += f"\n{args.guideline_code_prefix}{rewritten_guideline_index_for_current_prompt + 1}: {guideline.name}. "

        if args.llama_guard_prompt_template.should_include_guideline_descriptions:
            full_guidelines_text += f"\n{guideline.description}"

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
    eval_row: EvalRow,
    args: DatagenArgs,
    guideline_indices_to_include_in_llama_guard_prompt: List[int],
) -> str:
    to_return = eval_row.label

    if eval_row.label == "bad" and args.prompt_template.requires_fine_labels:
        original_violated_guideline_indices = set(
            convert_guideline_codes_to_indices(
                eval_row.violated_guideline_codes, args.guideline_code_prefix
            )
        )

        map_of_original_guideline_indices_to_rewritten_guideline_codes = get_map_of_original_guideline_indices_to_rewritten_guideline_codes(args, guideline_indices_to_include_in_llama_guard_prompt)

        rewritten_violated_guideline_codes = [map_of_original_guideline_indices_to_rewritten_guideline_codes[original_violated_index] for original_violated_index in original_violated_guideline_indices]

        to_return += "\n"
        to_return += ','.join(rewritten_violated_guideline_codes)

    if args.explanation_position_in_llama_guard_generation is not None:
        if args.explanation_position_in_llama_guard_generation == ExplanationPosition.PRE:
            to_return = f"Explanation: {eval_row.explanation}\n{to_return}"
        elif args.explanation_position_in_llama_guard_generation == ExplanationPosition.POST:
            to_return = f"{to_return}\nExplanation: {eval_row[args.explanation_key]}"

    return to_return

def get_map_of_original_guideline_indices_to_rewritten_guideline_codes(
    args: DatagenArgs,
    guideline_indices_to_include_in_llama_guard_prompt: List[int]
):
    return {
        original_guideline_index: args.guideline_code_prefix + str(rewritten_guideline_index + 1)
        for rewritten_guideline_index, original_guideline_index in enumerate(
            guideline_indices_to_include_in_llama_guard_prompt
        )
    }

def maybe_augment_with_good_row_with_empty_generation(
    eval_row: EvalRow,
    sft_rows_to_build: list[dict[str, str]],
    indices_of_all_guidelines: range,
    args: DatagenArgs,
) -> None:
    if (
        random.random()
        < args.probability_to_augment_with_good_rows_with_empty_generation
    ):
        eval_row_copy = copy.deepcopy(eval_row)
        eval_row_copy.generation = ""
        eval_row_copy.label = "safe"
        eval_row_copy.violated_guideline_codes = []

        sft_rows_to_build.append(
            make_sft_row(
                eval_row_copy,
                args,
                guideline_indeces_to_include_in_llama_guard_prompt=list(
                    indices_of_all_guidelines
                ),
            )
        )


def maybe_augment_with_rows_with_dropped_prompt_guidelines(
    eval_row: EvalRow,
    sft_rows_to_build: list[dict[str, str]],
    indices_of_all_guidelines: range,
    args: DatagenArgs,
) -> None:
    violated_guideline_indices = convert_guideline_codes_to_indices(
        eval_row.violated_guideline_codes, args.guideline_code_prefix
    )

    nonviolated_guideline_indices = list(
        set(indices_of_all_guidelines) - set(violated_guideline_indices)
    )

    maybe_augment_with_row_with_dropped_nonviolated_prompt_guidelines(
        eval_row,
        sft_rows_to_build,
        indices_of_all_guidelines,
        nonviolated_guideline_indices,
        args,
    )

    maybe_augment_with_row_with_dropped_violated_and_nonviolated_prompt_guidelines(
        eval_row,
        sft_rows_to_build,
        indices_of_all_guidelines,
        violated_guideline_indices,
        nonviolated_guideline_indices,
        args,
    )


def convert_guideline_codes_to_indices(codes: list[str], code_prefix: str) -> list[int]:
    # The string indexing of GLs is 1-based, so we convert to 0-based here for consistency with python indexing
    return [int(code.lstrip(code_prefix)) - 1 for code in codes]


def maybe_augment_with_row_with_dropped_nonviolated_prompt_guidelines(
    eval_row: EvalRow,
    sft_rows_to_build: list[dict[str, str]],
    indices_of_all_guidelines: range,
    nonviolated_guideline_indices: list[int],
    args: DatagenArgs,
) -> None:
    if not args.should_augment_with_rows_with_dropped_nonviolated_prompt_guidelines:
        pass

    number_of_guidelines_to_drop = random.randint(0, len(nonviolated_guideline_indices))

    if number_of_guidelines_to_drop == len(indices_of_all_guidelines):
        number_of_guidelines_to_drop -= 1

    dropped_guideline_indices = random.sample(
        nonviolated_guideline_indices, number_of_guidelines_to_drop
    )

    retained_guideline_indices = list(
        set(indices_of_all_guidelines) - (set(dropped_guideline_indices))
    )

    sft_rows_to_build.append(
        make_sft_row(
            eval_row,
            args,
            guideline_indeces_to_include_in_llama_guard_prompt=retained_guideline_indices,
        )
    )


def maybe_augment_with_row_with_dropped_violated_and_nonviolated_prompt_guidelines(
    eval_row: EvalRow,
    sft_rows_to_build: list[dict[str, str]],
    indices_of_all_guidelines: range,
    violated_guideline_indices: list[int],
    nonviolated_guideline_indices: list[int],
    args: DatagenArgs,
) -> None:
    if (
        eval_row.label == "safe"
        or not args.should_augment_with_rows_with_dropped_violated_and_nonviolated_prompt_guidelines
    ):
        pass

    random_nonviolated_guideline_indices_to_drop = random.sample(
        nonviolated_guideline_indices,
        random.randint(0, len(nonviolated_guideline_indices) - 1),
    )

    set_of_retained_guideline_indices = (
        set(indices_of_all_guidelines)
        - set(violated_guideline_indices)
        - set(random_nonviolated_guideline_indices_to_drop)
    )

    eval_row_copy = copy.deepcopy(eval_row)
    eval_row_copy.label = "safe"
    eval_row_copy.violated_guideline_codes = []

    sft_rows_to_build.append(
        make_sft_row(
            eval_row_copy,
            args,
            guideline_indeces_to_include_in_llama_guard_prompt=list(
                set_of_retained_guideline_indices
            ),
        )
    )
