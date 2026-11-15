"""Run a hallucination-detection method on a HaluEval split and log the results.

To switch dataset, change the task folder in INPUT_FILE ("qa", "dialogue" or "summarization") or the
file name ("pilot.jsonl" or "test.jsonl"). Each run writes its log to:

    runs/<method>/<date>/<date>_<time>_<task>_<split>.log      e.g. runs/fewl/2026-11-03/2026-11-03_14-05-09_qa_pilot.log

Run it with Debug in VS Code or `python3 run.py`.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from embeddings import get_embeddings
from fewl import find_neighbour_questions, gather_evidence, score_answer
from DatasetRetrievers.HaluEvaldataset import read_examples
from llm import ENV_FILE, load_api_key

# Here we keep every setting in one place, so it is easy to see and change.
INPUT_FILE = Path(__file__).parent / "HaluEvalData" / "qa" / "sample_1000.jsonl"
METHOD_NAME = "fewl"
RUNS_FOLDER = Path(__file__).parent / "runs"
QUESTIONS_AT_ONCE = 5   # how many questions are worked on at the same time; 1 runs them one after another

# Here we name, for each HaluEval task, which field is the question and which are the two answers.
TASK_FIELDS = {
    "qa": {"question": "question", "right": "right_answer", "hallucinated": "hallucinated_answer"},
    "dialogue": {"question": "dialogue_history", "right": "right_response", "hallucinated": "hallucinated_response"},
    "summarization": {"question": "document", "right": "right_summary", "hallucinated": "hallucinated_summary"},
}

# FEWL paper, Table 10: accuracy (%) on HaluEval QA with GPT-3.5 as the reference model.
PAPER_TABLE_10 = {"single_no_penalty": 94.33, "single_with_penalty": 97.89, "multi_no_penalty": 95.52, "fewl": 98.15}


def build_log_path(method_name, input_file):
    """Build the log path runs/<method>/<date>/<date>_<time>_<task>_<split>.log for a run starting now."""
    now = datetime.now()
    date_text = now.strftime("%Y-%m-%d")
    time_text = now.strftime("%H-%M-%S")

    # Here we name the file after the task folder (e.g. "qa") and the split file (e.g. "pilot").
    task_name = input_file.parent.name
    file_name = f"{date_text}_{time_text}_{task_name}_{input_file.stem}.log"
    return RUNS_FOLDER / method_name / date_text / file_name


def log_message(message, log_path):
    """Print a message and add it, with the current time, to the run's log file."""
    print(message)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    time_text = datetime.now().strftime("%H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(f"{time_text}  {message}\n")


def read_question_pool(raw_file_path, question_field):
    """Collect every distinct question in a task's full raw file, to search for neighbour questions in."""
    question_pool = []
    seen_questions = set()
    for example in read_examples(raw_file_path):
        question = example[question_field]
        # Here we skip repeats, so a question cannot be picked twice as a neighbour.
        if question not in seen_questions:
            question_pool.append(question)
            seen_questions.add(question)
    return question_pool


def build_embedding_table(question_pool, pool_embeddings):
    """Pair each pool question with its embedding, so any question's embedding can be looked up by its text."""
    embedding_table = {}
    for question, embedding in zip(question_pool, pool_embeddings):
        embedding_table[question] = embedding
    return embedding_table


def find_all_neighbours(examples, task_fields, embedding_table, question_pool, pool_embeddings):
    """Find the neighbour questions of every example once, before any model is asked anything."""
    neighbour_table = {}
    for example in examples:
        question = example[task_fields["question"]]

        # Here we look the embedding up instead of asking OpenAI again; every input question is in the pool.
        question_embedding = embedding_table[question]
        neighbour_table[question] = find_neighbour_questions(question_embedding, question_pool, pool_embeddings)
    return neighbour_table


def evaluate_example(example, task_fields, neighbour_questions, api_key):
    """Run FEWL on one example and score both its right and its hallucinated answer."""
    question = example[task_fields["question"]]
    evidence = gather_evidence(question, neighbour_questions, api_key)

    right_answer_scores = score_answer(example[task_fields["right"]], evidence, api_key)
    hallucinated_answer_scores = score_answer(example[task_fields["hallucinated"]], evidence, api_key)
    return {
        "right_answer_scores": right_answer_scores,
        "hallucinated_answer_scores": hallucinated_answer_scores,
        "evidence": evidence,
    }


def evaluate_examples_in_parallel(examples, task_fields, neighbour_table, api_key):
    """Run FEWL on QUESTIONS_AT_ONCE examples at a time, handing back each result in question order."""

    def evaluate_one(example):
        """Run FEWL on one example, using the neighbours found beforehand."""
        neighbour_questions = neighbour_table[example[task_fields["question"]]]
        return evaluate_example(example, task_fields, neighbour_questions, api_key)

    # Here each worker thread takes the next example as soon as it is free; while one waits for OpenAI,
    # the others keep going. map() hands the results back in the same order as the examples.
    with ThreadPoolExecutor(max_workers=QUESTIONS_AT_ONCE) as pool:
        for result in pool.map(evaluate_one, examples):
            yield result


def compute_accuracy(results):
    """For each scoring method, the percent of examples where the right answer scored strictly higher."""
    accuracy = {}
    for method_name in results[0]["right_answer_scores"]:
        number_correct = 0
        for result in results:
            # Here we check whether this method ranked the right answer above the hallucinated one.
            if result["right_answer_scores"][method_name] > result["hallucinated_answer_scores"][method_name]:
                number_correct += 1
        accuracy[method_name] = 100 * number_correct / len(results)
    return accuracy


def log_accuracy_table(accuracy, number_of_examples, log_path):
    """Log the accuracy of each scoring method next to the paper's Table 10 value."""
    log_message(f"Accuracy on {number_of_examples} examples (right answer scored higher than hallucinated):", log_path)
    for method_name, value in accuracy.items():
        paper_value = PAPER_TABLE_10.get(method_name)
        paper_text = f"{paper_value:.2f}%" if paper_value else "not reported"
        log_message(f"  {method_name:22} {value:6.2f}%   paper (HaluEval QA): {paper_text}", log_path)


def run_fewl(input_file, log_path):
    """Run FEWL on every example in the input file, logging each result and the final accuracy."""
    # Here we load the API key and the settings for this task; step over these lines to check them.
    api_key = load_api_key(ENV_FILE)
    task_name = input_file.parent.name
    task_fields = TASK_FIELDS[task_name]
    raw_file_path = input_file.parent.parent / "raw" / f"{task_name}_data.json"
    log_message(f"Method: fewl | input: {input_file} | neighbour pool: {raw_file_path}", log_path)

    # Here we embed every question in the full task file once; later runs read them from the cache.
    question_pool = read_question_pool(raw_file_path, task_fields["question"])
    pool_embeddings = get_embeddings(question_pool, api_key)
    log_message(f"Embedded {len(question_pool)} pool questions", log_path)

    # Here we find every example's neighbours up front, using the embeddings we already have.
    examples = read_examples(input_file)
    embedding_table = build_embedding_table(question_pool, pool_embeddings)
    neighbour_table = find_all_neighbours(examples, task_fields, embedding_table, question_pool, pool_embeddings)
    log_message(f"Found neighbours for {len(neighbour_table)} questions", log_path)

    # Here we log each example as soon as its result is ready, in question order.
    results = []
    results_in_order = evaluate_examples_in_parallel(examples, task_fields, neighbour_table, api_key)
    for example_number, result in enumerate(results_in_order, start=1):
        example = examples[example_number - 1]
        results.append(result)

        right_score = result["right_answer_scores"]["fewl"]
        hallucinated_score = result["hallucinated_answer_scores"]["fewl"]
        log_message(f"[{example_number}/{len(examples)}] fewl right={right_score:.4f} "
                    f"hallucinated={hallucinated_score:.4f} | {example[task_fields['question']][:80]}", log_path)
        # Here we also log the full result as one JSON line, so it can be inspected or re-read later.
        log_message(f"details: {json.dumps(result, ensure_ascii=False)}", log_path)

    log_accuracy_table(compute_accuracy(results), len(results), log_path)
    return results

# Full log fucntions-----------
def log_example_in_full(example_number, number_of_examples, example, task_fields, evidence, log_path):
    """Write everything FEWL used for one example: its question, test answers, and all the evidence."""
    log_message(f"===== Example {example_number}/{number_of_examples} =====", log_path)
    log_message(f"Question: {evidence['question']}", log_path)
    log_message(f"Test answer 1 (right): {example[task_fields['right']]}", log_path)
    log_message(f"Test answer 2 (hallucinated): {example[task_fields['hallucinated']]}", log_path)

    reference_answers = evidence["reference_answers"]
    log_message(f"Reference answers ({len(reference_answers)}):", log_path)
    for answer_number, answer in enumerate(reference_answers, start=1):
        log_message(f"  {answer_number}. {answer}", log_path)

    pairs = evidence["wrong_and_corrected_pairs"]
    log_message(f"Wrong/corrected pairs ({len(pairs)}):", log_path)
    for pair_number, pair in enumerate(pairs, start=1):
        log_message(f"  {pair_number}. Wrong: {pair['wrong']}", log_path)
        log_message(f"     Corrected: {pair['corrected']}", log_path)

    neighbour_questions = evidence["neighbour_questions"]
    log_message(f"Neighbour questions ({len(neighbour_questions)}), each with its answers:", log_path)
    for neighbour_index, neighbour_question in enumerate(neighbour_questions):
        log_message(f"  {neighbour_index + 1}. {neighbour_question}", log_path)

        # Here we pick the answers that belong to this neighbour question; both lists are in the same order.
        neighbour_answers = evidence["neighbour_reference_answers"][neighbour_index]
        for answer_number, answer in enumerate(neighbour_answers, start=1):
            log_message(f"     Answer {answer_number}: {answer}", log_path)


def prepare_fewl_run(input_file, log_path):
    """Load what a FEWL run needs: the API key, the task's field names, and the embedded question pool."""
    # Here we load the API key and the settings for this task; step over these lines to check them.
    api_key = load_api_key(ENV_FILE)
    task_name = input_file.parent.name
    task_fields = TASK_FIELDS[task_name]
    raw_file_path = input_file.parent.parent / "raw" / f"{task_name}_data.json"
    log_message(f"Method: fewl | input: {input_file} | neighbour pool: {raw_file_path}", log_path)

    # Here we embed every question in the full task file once; later runs read them from the cache.
    question_pool = read_question_pool(raw_file_path, task_fields["question"])
    pool_embeddings = get_embeddings(question_pool, api_key)
    log_message(f"Embedded {len(question_pool)} pool questions", log_path)
    return api_key, task_fields, question_pool, pool_embeddings


def run_fewl_full_log(input_file, log_path):
    """Run FEWL on every example in the input file, logging all its evidence and the final accuracy."""
    api_key, task_fields, question_pool, pool_embeddings = prepare_fewl_run(input_file, log_path)

    # Here we find every example's neighbours up front, using the embeddings we already have.
    examples = read_examples(input_file)
    embedding_table = build_embedding_table(question_pool, pool_embeddings)
    neighbour_table = find_all_neighbours(examples, task_fields, embedding_table, question_pool, pool_embeddings)
    log_message(f"Found neighbours for {len(neighbour_table)} questions", log_path)

    # Here we log each example as soon as its result is ready, in question order.
    results = []
    results_in_order = evaluate_examples_in_parallel(examples, task_fields, neighbour_table, api_key)
    for example_number, result in enumerate(results_in_order, start=1):
        example = examples[example_number - 1]
        results.append(result)
        log_example_in_full(example_number, len(examples), example, task_fields, result["evidence"], log_path)

        # Here we finish the example's section with what FEWL decided, using the evidence listed above.
        right_score = result["right_answer_scores"]["fewl"]
        hallucinated_score = result["hallucinated_answer_scores"]["fewl"]
        log_message(f"FEWL score: right={right_score:.4f}, hallucinated={hallucinated_score:.4f}\n", log_path)

    log_accuracy_table(compute_accuracy(results), len(results), log_path)
    return results

# End of full log functions--------

def main():
    """Run the method named in METHOD_NAME on INPUT_FILE."""
    log_path = build_log_path(METHOD_NAME, INPUT_FILE)
    run_fewl(INPUT_FILE, log_path)
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
