"""Download the HaluEval benchmark, split each task into a pilot set and a test set, and save a random sample.

Run it with Debug in VS Code or `python3 HaluEvaldataset.py`. Everything is saved in HaluEvalData/ at the repo root:

    HaluEvalData/raw/qa_data.json      the original file, exactly as downloaded
    HaluEvalData/qa/pilot.jsonl        a small random sample for quick experiments
    HaluEvalData/qa/test.jsonl         everything else, kept for the real evaluation
    HaluEvalData/qa/sample_1000.jsonl  SAMPLE_SIZE random examples from the full raw file

All three come from one shuffle of the raw file (RANDOM_SEED), so the pilot is the first rows of every
sample, and a bigger sample always contains a smaller one.
"""

import json
import random
import urllib.request
from pathlib import Path

# Here we keep every setting in one place, so it is easy to see and change.
SOURCE_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/{task_name}_data.json"
TASK_NAMES = ["qa", "dialogue", "summarization", "general"]
PILOT_FRACTION = 0.001
SAMPLE_SIZE = 100
RANDOM_SEED = 42
DATA_FOLDER = Path(__file__).parent.parent / "HaluEvalData"


def download_task_file(task_name, raw_folder):
    """Download one HaluEval task file into the raw folder and return where it was saved."""
    file_path = raw_folder / f"{task_name}_data.json"

    # Here we skip the download if the file is already on disk, so running the script again is fast.
    if file_path.exists():
        print(f"{task_name}: already downloaded")
        return file_path

    url = SOURCE_URL.format(task_name=task_name)
    print(f"{task_name}: downloading {url}")
    urllib.request.urlretrieve(url, file_path)
    return file_path


def read_examples(file_path):
    """Read a file that has one JSON example per line and return the examples as a list."""
    examples = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            # Here we turn one line of text into a Python dictionary.
            examples.append(json.loads(line))
    return examples


def shuffle_examples(examples, random_seed):
    """Return a shuffled copy of the examples, always in the same order for the same seed."""
    shuffled_examples = examples.copy()

    # Here we shuffle with a fixed seed, so every run produces exactly the same order.
    random.Random(random_seed).shuffle(shuffled_examples)
    return shuffled_examples


def split_pilot_and_test(examples, pilot_fraction, random_seed):
    """Shuffle the examples and cut them into a small pilot set and a test set that do not overlap."""
    shuffled_examples = shuffle_examples(examples, random_seed)

    # Here we take the first slice as the pilot set and keep the rest as the test set.
    pilot_size = round(len(shuffled_examples) * pilot_fraction)
    pilot_examples = shuffled_examples[:pilot_size]
    test_examples = shuffled_examples[pilot_size:]
    return pilot_examples, test_examples


def take_random_sample(examples, sample_size, random_seed):
    """Shuffle the examples and keep the first sample_size of them (or all, if there are fewer)."""
    shuffled_examples = shuffle_examples(examples, random_seed)
    return shuffled_examples[:sample_size]


def write_examples(examples, file_path):
    """Save the examples to a file, one JSON example per line."""
    with open(file_path, "w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def prepare_task(task_name, data_folder):
    """Download one HaluEval task, split it into pilot and test sets, take a random sample, and save all three."""
    raw_folder = data_folder / "raw"
    task_folder = data_folder / task_name
    raw_folder.mkdir(parents=True, exist_ok=True)
    task_folder.mkdir(parents=True, exist_ok=True)

    raw_file_path = download_task_file(task_name, raw_folder)
    examples = read_examples(raw_file_path)
    pilot_examples, test_examples = split_pilot_and_test(examples, PILOT_FRACTION, RANDOM_SEED)

    write_examples(pilot_examples, task_folder / "pilot.jsonl")
    write_examples(test_examples, task_folder / "test.jsonl")
    print(f"{task_name}: {len(examples)} examples -> {len(pilot_examples)} pilot, {len(test_examples)} test")

    sample_examples = take_random_sample(examples, SAMPLE_SIZE, RANDOM_SEED)
    write_examples(sample_examples, task_folder / f"sample_{SAMPLE_SIZE}.jsonl")
    print(f"{task_name}: {len(sample_examples)} examples saved to sample_{SAMPLE_SIZE}.jsonl")


def main():
    """Prepare every HaluEval task listed in TASK_NAMES."""
    for task_name in TASK_NAMES:
        prepare_task(task_name, DATA_FOLDER)
    print(f"Done. Files are in {DATA_FOLDER}")


if __name__ == "__main__":
    main()
