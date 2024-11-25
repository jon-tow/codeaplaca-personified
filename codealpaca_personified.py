"""CodeAlpaca-Personified based on: https://github.com/sahil280114/codealpaca"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import string
from datetime import datetime
from functools import partial
from tqdm import tqdm

import datasets
import database
import openai
import tqdm
import transformers
from datasketch import MinHash, MinHashLSH
from pydantic import Field
from rich import print
from rich.align import Align
from rich.panel import Panel
from rich.text import Text
from rouge_score import rouge_scorer
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Utils


def display(sender: str, message: str, color: str = "blue", raw: bool = False):
    if raw:
        print(message)
        return
    text = Text(message)
    panel = Panel(
        Align(text),
        expand=False,
        border_style=color,
        title=f"{sender}",
    )
    print(panel)


def lowercase_first_letter(text: str) -> str:
    return text[0].lower() + text[1:]


# Database Table


class AlpacaTurn(database.Table):
    instruction: str
    input: str = "None"
    output: str
    model: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


# CodeAlpaca Personified


PROMPT = """\
You are asked to come up with a set of 20 extremely diverse code generation task instructions. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
3. The type of instructions should be diverse. The list should include diverse types of programming tasks like open-ended generation, classification, editing, optimization etc.
2. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
3. The instructions should be in English.
4. The instructions should at least 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
5. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
6. Not all instructions require input. For example, when a instruction asks about some general information, "write a python program to load a file.", it is not necssary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
7. The output should be an appropriate response to the instruction and the input.
8. All tasks should be coding or programming related.
9. You must follow the format exactly as described in the prompt:
### Task i
i. Instruction: <instruction>
i. Input: <input>
i. Output: <output>
### End of Task i

List of 20 tasks:"""


def get_persona_prompt(persona_dataset: list[dict[str, str]]) -> str:
    return f"You are {lowercase_first_letter(random.choice(persona_dataset)['persona'].replace(' R ', ' Python ')).strip()}"


def encode_prompt(prompt_instructions: list[dict[str, str]]) -> str:
    """Encode multiple prompt instructions into a single string."""
    import re

    prompt = PROMPT
    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = (
            task_dict["instruction"],
            task_dict["input"],
            task_dict["output"],
        )
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"\n\n### Task {idx + 1}\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
        prompt += f"### End of Task {idx + 1}"
    prompt += f"\n\n### Task {idx + 2}\n{idx+2}. Instruction:"
    return prompt


def post_process_completion(
    num_prompt_instructions: int,
    response: str,
) -> list[dict[str, str]]:
    """Ref: https://github.com/sahil280114/codealpaca/blob/2f78ddc5c682ed6738ad092bbbfa59ba915afcb0/generate_instruction.py#L44"""
    if not response:
        return []
    # Parse out the instructions from the response
    num_new_tasks = range(num_prompt_instructions + 1, 21)
    raw_instructions = []
    for i in num_new_tasks:
        section = re.search(
            f"### Task {i}(.*?)### End of Task {i}", response, re.DOTALL
        )
        if section is not None:
            raw_instructions.append(section.group(1))
    logger.debug(f"Number of raw instructions: {len(raw_instructions)}")

    instructions = []
    for idx, inst in enumerate(raw_instructions):
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # Filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 175:
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit confusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # Filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # Filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def lsh_deduplicate(
    new_samples: list[dict[str, str]],
    existing_samples: list[dict[str, str]],
    threshold: float = 0.5,
    num_perm: int = 128,
) -> list[dict[str, str]]:
    def get_ngrams(text: str, n: int = 5) -> list[str]:
        tokens = text.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def create_minhash(text: str, num_perm: int = 128) -> MinHash:
        ngrams = get_ngrams(text)
        m = MinHash(num_perm=num_perm)
        for ngram in ngrams:
            m.update(ngram.encode("utf8"))
        return m

    # Initialize LSH for the MinHash signatures with the specified threshold
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Add existing database samples to LSH
    for idx, example in enumerate(tqdm(existing_samples, desc="Indexing Database")):
        text = example["instruction"]
        m = create_minhash(text, num_perm)
        # Insert with "db_" prefix to distinguish database entries
        lsh.insert(f"db_{idx}", m)

    unique_new_samples = []  # Store unique samples from new data

    for idx, example in enumerate(tqdm(new_samples, desc="Processing New Samples")):
        text = example["instruction"]
        m = create_minhash(text, num_perm)

        # Check if similar samples exist in the LSH index (database + previous new samples)
        result = lsh.query(m)
        if len(result) == 0:  # If no near-duplicates, add to unique_new_samples
            lsh.insert(f"new_{idx}", m)  # Insert new sample to LSH
            unique_new_samples.append(example)  # Mark it as unique

    return unique_new_samples


# Args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dataset", type=str, default="seeds.jsonl")
    parser.add_argument("--num_prompt_instructions", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--db_path", type=str, default=":memory:")
    parser.add_argument(
        "--dedup-method", type=str, default="lsh", choices=["lsh", "rouge"]
    )
    return parser.parse_args()


# Program


if __name__ == "__main__":
    # python codealpaca.py --db_path ".cache/test_codealpaca.db" --num_samples 1024 --batch_size 64
    args = parse_args()
    # Ensure num_samples is divisible by batch_size
    args.num_samples = args.num_samples - (args.num_samples % args.batch_size)

    # System setup
    hostname = os.uname().nodename.split(".")[0]
    filename = __file__.split("/")[-1][:-3]
    default_root_dir = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"hostname: `{hostname}`")
    logger.debug(f"filename: `{filename}`")
    logger.debug(f"root_dir: `{default_root_dir}`")

    # Initialize the database
    db_path = args.db_path
    logger.debug(f"db_path: `{db_path}`")
    if db_path != ":memory:":
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = database.Database(db_path)
    db.create_table(AlpacaTurn, "codealpaca")

    # Determine the number of samples left to generate if the database already has some
    num_samples_to_generate = max(0, args.num_samples - db.count("codealpaca"))
    logger.info(f"Number of samples to generate: {num_samples_to_generate}")

    # Initialize seed dataset
    seed_dataset = datasets.load_dataset("json", data_files=args.seed_dataset)["train"]
    seed_dataset = [
        {
            "instruction": d["instruction"],
            "input": d["instances"][0]["input"],
            "output": d["instances"][0]["output"],
        }
        for d in seed_dataset
    ]
    logger.info(f"Loaded {len(seed_dataset)} human-generated seed examples.")

    # Initialize persona dataset
    persona_dataset = datasets.load_dataset("jon-tow/FineProgrammingPersonas-v0.1")[
        "train"
    ]
    logger.info(f"Loaded {len(persona_dataset)} persona examples.")
    # Filter for personas that have 'Python' in them
    # persona_dataset = persona_dataset.filter(lambda x: "python" in x["persona"].lower(), num_proc=os.cpu_count() - 32)
    # logger.info(f"Filtered to {len(persona_dataset)} Python-related persona examples.")

    # Setup n-gram similarity scorer
    scorer = (
        rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        if args.dedup_method == "rouge"
        else None
    )

    # Initialize the model
    client = openai.OpenAI(base_url="http://localhost:8002/v1", api_key="EMPTY")
    model = client.models.list().data[0].id
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    # Loop until we insert the desired number of samples
    pbar = tqdm(total=num_samples_to_generate, desc="Generating...")
    total_inserted = 0
    while total_inserted < num_samples_to_generate:
        samples_remaining = num_samples_to_generate - total_inserted
        batch_size = min(args.batch_size, samples_remaining)

        # Generate prompts
        prompts = [
            encode_prompt(random.sample(seed_dataset, k=args.num_prompt_instructions))
            for _ in range(batch_size)
        ]
        display("Seed Prompt", prompts[0], color="green")

        # Randomly select sampling settings
        stop_delimiter = "### End of Task 20"
        sample_kwargs = dict(
            temperature=1.0,
            top_p=random.choice([0.95, 0.99]),
            max_tokens=16_384,
            stop=[stop_delimiter],
        )
        display("Sampling Kwargs", json.dumps(sample_kwargs, indent=2), color="blue")

        # Apply the chat template
        prompts = [
            tokenizer.apply_chat_template(
                [
                    # TODO: Quality of instruction/outputs pairs degrades when the persona
                    # prompt is fed into the system prompt... Why?
                    # {
                    #     "role": "system",
                    #     "content": f"{get_persona_prompt(persona_dataset)}",
                    # },
                    {
                        "role": "user",
                        "content": f"{get_persona_prompt(persona_dataset)}\n\n{p}",
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        display("Message", prompts[0], color="blue")

        # Generate the CodeAlpaca examples
        response = client.completions.create(
            model=model,
            prompt=prompts,
            **sample_kwargs,
        )
        completions = [c.text.strip() + f"\n{stop_delimiter}" for c in response.choices]
        display("Completions", completions[0], color="green")
        logger.debug(f"Completions:\n{completions[0]}")

        instructions = []
        for c in completions:
            instructions.extend(
                post_process_completion(args.num_prompt_instructions, c)
            )
        display("Instructions", instructions[0]["instruction"], color="green")

        # Compute similarity between new instructions and existing instructions in the database
        if db.count("codealpaca") > 0:

            final_instructions = []
            existing_instructions = db.sql("SELECT instruction FROM codealpaca")

            # Deduplicate
            # NOTE: The original codealpaca deduplication method is very slow via
            # rouge-score-based n-gram similarity
            if args.dedup_method == "lsh":
                final_instructions = lsh_deduplicate(
                    instructions, existing_instructions, threshold=0.1
                )
            elif args.dedup_method == "rouge":
                from multiprocessing import Pool

                all_instruction_tokens = [
                    scorer._tokenizer.tokenize(existing_instructions[j]["instruction"])
                    for j in range(len(existing_instructions))
                ]
                for i in tqdm(
                    range(len(instructions)), desc="Filtering by similarity..."
                ):
                    # Compute similarity with the pre-tokenized instructions
                    new_instruction_tokens = scorer._tokenizer.tokenize(
                        instructions[i]["instruction"]
                    )
                    with Pool(os.cpu_count() - 32) as p:
                        rouge_scores = p.map(
                            partial(rouge_scorer._score_lcs, new_instruction_tokens),
                            all_instruction_tokens,
                        )
                    rouge_scores = [score.fmeasure for score in rouge_scores]
                    if i == 0:
                        logger.debug(
                            f"Top rouge scores: {sorted(rouge_scores, reverse=True)[:5]}"
                        )

                    # Filter out instructions that are too similar to existing instructions
                    if max(rouge_scores) > 0.7:
                        display(
                            f"Skipping Similar Instruction {i} ({max(rouge_scores):.3f})",
                            instructions[i]["instruction"],
                            color="red",
                        )
                        continue
                    all_instruction_tokens.append(new_instruction_tokens)
                    final_instructions.append(instructions[i])

            instructions = final_instructions

        # Create examples for the database
        examples = [
            AlpacaTurn(
                instruction=ins["instruction"],
                input=ins["input"],
                output=ins["output"],
                model=model,
                id=str(
                    hashlib.sha256(
                        (ins["instruction"] + ins["input"] + ins["output"]).encode()
                    ).hexdigest()
                ),
            )
            for ins in instructions
        ]

        # Add examples to the database...
        added = db.add("codealpaca", examples)
        logger.info(f"Added {len(added)} examples to the database")
        total_inserted += len(added)
        pbar.update(len(added))

    pbar.close()

    os.makedirs(os.path.join(default_root_dir, "data"), exist_ok=True)
    db.write_to_jsonl(
        "codealpaca",
        os.path.join(
            default_root_dir,
            "data",
            f"codealpaca_personified_{args.num_samples}.jsonl",
        ),
    )
