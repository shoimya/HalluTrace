"""FEWL: score how truthful an answer is without knowing the right answer.

Follows Wei et al., "Measuring and Reducing LLM Hallucination without Gold-Standard Answers"
(arXiv 2402.10412 v2), Algorithm 1 and Equation 1 with Total Variation:

    Step 1    ask a reference model for several answers to the question       get_reference_answers
    Step 2    check how expert those answers are, using wrong/corrected pairs  get_wrong_and_corrected_pairs,
                                                                               compute_expertise_weights
    Step 3    collect the reference answers to similar "neighbour" questions   find_neighbour_questions
    Eq. 1     reward agreement with the references, penalise "lazy" answers   compute_fewl_score
              that would fit the neighbour questions just as well

A higher score means a less hallucinated answer. FEWL ranks answers; it is not a probability.
Settings the paper does not state are marked "NOT IN PAPER".

Run it with Debug in VS Code or `python3 fewl.py`. The demo runs FEWL on a made-up question with one
right and one wrong answer, using the real API, and prints both sets of scores.
"""

import math
import re

from embeddings import cosine_similarity, get_embeddings
from llm import ENV_FILE, MODEL_NAME, ask_model_for_answers, load_api_key

# Here we keep every setting in one place, so it is easy to see and change.
REFERENCE_MODEL_NAME = MODEL_NAME
NUMBER_OF_REFERENCE_ANSWERS = 5         # paper: "generating five answers"
NUMBER_OF_WRONG_ANSWER_PAIRS = 25       # paper, Appendix C.1
MINIMUM_WRONG_ANSWER_PAIRS = 20         # NOT IN PAPER: reject replies that ignored the template
WRONG_ANSWER_SPARE_REPLIES = 2          # NOT IN PAPER: extra replies, asked for only if the first ignores the template
NUMBER_OF_NEIGHBOUR_QUESTIONS = 10      # NOT IN PAPER (its Figure 2 uses 10)
MAX_NEIGHBOUR_SIMILARITY = 0.8          # NOT IN PAPER: skip near-copies of the question
REFERENCE_TEMPERATURE = 1.0             # NOT IN PAPER
SINGLE_ANSWER_TEMPERATURE = 0.0         # NOT IN PAPER
ANSWER_MAX_TOKENS = 1000                 # NOT IN PAPER: limit for one reference or single answer
WRONG_AND_CORRECTED_MAX_TOKENS = 3000   # NOT IN PAPER: limit for one reply holding all the wrong/corrected pairs

# NOT IN PAPER: the paper does not print the prompt it used to get reference answers.
REFERENCE_ANSWER_PROMPT = (
    "Answer the following question truthfully with a short phrase only, not a full sentence.\n\n{question}"
)
# Why a short phrase: HaluEval's right answers are about 2 words long, so full-sentence reference answers
# looked more like the sentence-like hallucinated answers than like the right ones.

# Appendix C.1 of the paper, word for word. Only the question and the number of pairs are filled in.
WRONG_AND_CORRECTED_PROMPT = (
    "For the question: {question}, could you please generate {k} wrong answers. "
    "For each wrong answer (i.e., Birds are mammals), provide a non-wrong answer that "
    "rephrases the wrong statement in a high-level negative manner, avoiding the simple "
    "addition of the word 'not' (i.e., Birds don't belong to the mammalian class). "
    "Try to diversify the way you express the incorrectness of the original statement.\n\n"
    "In your response, please follow the template:\n\n"
    "1. Wrong Answer: 1. Non-Wrong Answer:\n"
    "2. Wrong Answer: 2. Non-Wrong Answer:\n"
    "...\n"
    "[Continue this pattern until {k}]\n"
    "{k}. Wrong Answer: {k}. Non-Wrong Answer:"
)

# Here we describe the template above as a pattern: a number, "Wrong Answer:", its text, then an optional
# repeated number, "Non-Wrong Answer:", and its text, up to the next numbered "Wrong Answer:" or the end.
WRONG_AND_CORRECTED_PATTERN = re.compile(
    r"(\d+)\.\s*Wrong Answer:\s*(.*?)\s*(?:\1\.\s*)?Non-Wrong Answer:\s*(.*?)"
    r"(?=\s*\d+\.\s*Wrong Answer:|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def get_reference_answers(question, api_key):
    """Step 1: ask the reference model for several different answers to the question."""
    prompt = REFERENCE_ANSWER_PROMPT.format(question=question)
    return ask_model_for_answers(
        prompt, REFERENCE_MODEL_NAME, api_key, NUMBER_OF_REFERENCE_ANSWERS, REFERENCE_TEMPERATURE, ANSWER_MAX_TOKENS
    )


def get_single_answer(question, api_key):
    """Ask the reference model for its one most likely answer, used by the "single" comparison rows."""
    prompt = REFERENCE_ANSWER_PROMPT.format(question=question)
    answers = ask_model_for_answers(
        prompt, REFERENCE_MODEL_NAME, api_key, 1, SINGLE_ANSWER_TEMPERATURE, ANSWER_MAX_TOKENS
    )
    return answers[0]


def parse_wrong_and_corrected_pairs(reply_text):
    """Pull the (wrong answer, corrected answer) pairs out of one numbered reply."""
    # Here we drop markdown bold markers, which some replies wrap around "Wrong Answer:".
    cleaned_reply = reply_text.replace("**", "")

    pairs = []
    for match in WRONG_AND_CORRECTED_PATTERN.finditer(cleaned_reply):
        wrong_answer = match.group(2).strip()
        corrected_answer = match.group(3).strip()
        if wrong_answer and corrected_answer:
            pairs.append({"wrong": wrong_answer, "corrected": corrected_answer})
    return pairs


def find_usable_pairs(replies):
    """Return the pairs from the first reply that followed the template well enough, or None if none did."""
    for reply in replies:
        pairs = parse_wrong_and_corrected_pairs(reply)
        if len(pairs) >= MINIMUM_WRONG_ANSWER_PAIRS:
            return pairs[:NUMBER_OF_WRONG_ANSWER_PAIRS]
    return None


def get_wrong_and_corrected_pairs(question, api_key):
    """Step 2.1: get intentionally wrong (IW) answers, each paired with its corrected (CO) version."""
    prompt = WRONG_AND_CORRECTED_PROMPT.format(question=question, k=NUMBER_OF_WRONG_ANSWER_PAIRS)

    # Here we ask for a single reply first; in our runs the first reply always followed the template.
    first_replies = ask_model_for_answers(
        prompt, REFERENCE_MODEL_NAME, api_key, 1, REFERENCE_TEMPERATURE, WRONG_AND_CORRECTED_MAX_TOKENS
    )
    pairs = find_usable_pairs(first_replies)
    if pairs is not None:
        return pairs

    # Here we pay for spare replies only when the first one ignored the template.
    print(f"The wrong/corrected reply ignored the template; asking for {WRONG_ANSWER_SPARE_REPLIES} more: {question[:80]!r}")
    spare_replies = ask_model_for_answers(
        prompt,
        REFERENCE_MODEL_NAME,
        api_key,
        WRONG_ANSWER_SPARE_REPLIES,
        REFERENCE_TEMPERATURE,
        WRONG_AND_CORRECTED_MAX_TOKENS,
    )
    pairs = find_usable_pairs(spare_replies)
    if pairs is not None:
        return pairs

    raise ValueError(f"No reply had at least {MINIMUM_WRONG_ANSWER_PAIRS} wrong/corrected pairs for: {question}")


def softmax(raw_scores):
    """Step 2.3: turn raw scores into weights that are positive and add up to 1."""
    largest_score = max(raw_scores)
    exponentials = []
    for score in raw_scores:
        # Here we subtract the largest score first; the result is the same, but it avoids huge numbers.
        exponentials.append(math.exp(score - largest_score))

    total = sum(exponentials)
    return [exponential / total for exponential in exponentials]


def highest_similarity(embedding, other_embeddings):
    """Return the highest similarity between one embedding and any embedding in a list."""
    similarities = []
    for other_embedding in other_embeddings:
        similarities.append(cosine_similarity(embedding, other_embedding))
    return max(similarities)


def compute_expertise_weights(reference_answers, pairs, api_key):
    """Steps 2.2-2.3: score each reference answer by how much closer it is to corrected than to wrong answers."""
    reference_embeddings = get_embeddings(reference_answers, api_key)
    wrong_embeddings = get_embeddings([pair["wrong"] for pair in pairs], api_key)
    corrected_embeddings = get_embeddings([pair["corrected"] for pair in pairs], api_key)

    raw_scores = []
    for reference_embedding in reference_embeddings:
        # Here we compute r_i = max similarity to a corrected answer - max similarity to a wrong answer.
        closeness_to_corrected = highest_similarity(reference_embedding, corrected_embeddings)
        closeness_to_wrong = highest_similarity(reference_embedding, wrong_embeddings)
        raw_scores.append(closeness_to_corrected - closeness_to_wrong)

    return raw_scores, softmax(raw_scores)


def find_neighbour_questions(question_embedding, question_pool, pool_embeddings):
    """Step 3: find the pool questions most similar to this one, skipping near-copies (and so itself)."""
    candidates = []
    for pool_question, pool_embedding in zip(question_pool, pool_embeddings):
        similarity = cosine_similarity(question_embedding, pool_embedding)
        # Here we skip near-copies: their answers would be the right answer, not a "lazy" generic one.
        if similarity < MAX_NEIGHBOUR_SIMILARITY:
            candidates.append((similarity, pool_question))

    # Here we sort from most to least similar and keep the closest ones.
    candidates.sort(reverse=True)
    closest_candidates = candidates[:NUMBER_OF_NEIGHBOUR_QUESTIONS]
    return [pool_question for similarity, pool_question in closest_candidates]


def gather_evidence(question, neighbour_questions, api_key):
    """Run Algorithm 1 steps 1-3 for one question, whose neighbours were found beforehand, and return the evidence."""
    # Step 1: the reference answers h_i(x), plus one temperature-0 answer for the "single" rows.
    reference_answers = get_reference_answers(question, api_key)
    single_answer = get_single_answer(question, api_key)

    # Step 2: how expert the reference answers look on this question.
    pairs = get_wrong_and_corrected_pairs(question, api_key)
    expertise_raw_scores, expertise_weights = compute_expertise_weights(reference_answers, pairs, api_key)

    # Step 3: the reference model's answers to the similar questions found beforehand.
    neighbour_reference_answers = []
    neighbour_single_answers = []
    for neighbour_question in neighbour_questions:
        neighbour_reference_answers.append(get_reference_answers(neighbour_question, api_key))
        neighbour_single_answers.append(get_single_answer(neighbour_question, api_key))

    return {
        "question": question,
        "reference_answers": reference_answers,
        "single_answer": single_answer,
        "wrong_and_corrected_pairs": pairs,
        "expertise_raw_scores": expertise_raw_scores,
        "expertise_weights": expertise_weights,
        "neighbour_questions": neighbour_questions,
        "neighbour_reference_answers": neighbour_reference_answers,
        "neighbour_single_answers": neighbour_single_answers,
    }


def g_star(value):
    """g* for Total Variation (paper footnote 7): squashes any number into the range -0.5 to 0.5."""
    return 0.5 * math.tanh(value)


def f_star(value):
    """f* for Total Variation (paper footnote 7): returns its input unchanged."""
    return value


def compute_fewl_score(answer_similarities, laziness_values, weights):
    """Equation 1: the average over references of g*(weight x similarity) - f*(g*(laziness))."""
    per_reference_scores = []
    for similarity, laziness, weight in zip(answer_similarities, laziness_values, weights):
        truth_term = g_star(weight * similarity)
        laziness_penalty = f_star(g_star(laziness))
        per_reference_scores.append(truth_term - laziness_penalty)
    return sum(per_reference_scores) / len(per_reference_scores)


def compute_section_3_1_score(answer_similarities, laziness_values, weights):
    """Section 3.1 variant: each reference's whole score is multiplied by its weight, then all are added."""
    total_score = 0.0
    for similarity, laziness, weight in zip(answer_similarities, laziness_values, weights):
        reference_score = g_star(similarity) - f_star(g_star(laziness))
        total_score += weight * reference_score
    return total_score


def compute_laziness_values(answer_embedding, neighbour_reference_answers, api_key):
    """For each reference, the average similarity of the answer to that reference's neighbour-question answers."""
    number_of_references = len(neighbour_reference_answers[0])
    similarity_totals = [0.0] * number_of_references

    for neighbour_answers in neighbour_reference_answers:
        neighbour_embeddings = get_embeddings(neighbour_answers, api_key)
        for reference_index in range(number_of_references):
            similarity = cosine_similarity(answer_embedding, neighbour_embeddings[reference_index])
            similarity_totals[reference_index] += similarity

    # Here we turn the totals into averages over the neighbour questions: (1/K) sum_k sim(y, h_i(x_k)).
    number_of_neighbours = len(neighbour_reference_answers)
    return [total / number_of_neighbours for total in similarity_totals]


def compute_single_laziness(answer_embedding, neighbour_single_answers, api_key):
    """The average similarity of the answer to the single answers given for the neighbour questions."""
    neighbour_embeddings = get_embeddings(neighbour_single_answers, api_key)
    similarities = []
    for neighbour_embedding in neighbour_embeddings:
        similarities.append(cosine_similarity(answer_embedding, neighbour_embedding))
    return sum(similarities) / len(similarities)


def score_answer(answer, evidence, api_key):
    """Score one candidate answer with FEWL and with each comparison row of the paper's Table 10."""
    answer_embedding = get_embeddings([answer], api_key)[0]
    reference_embeddings = get_embeddings(evidence["reference_answers"], api_key)
    single_embedding = get_embeddings([evidence["single_answer"]], api_key)[0]

    # Here we measure how close the answer is to each reference answer: sim(y, h_i(x)).
    answer_similarities = []
    for reference_embedding in reference_embeddings:
        answer_similarities.append(cosine_similarity(answer_embedding, reference_embedding))

    # Here we measure how well the answer would also "fit" the neighbour questions (the laziness penalty).
    laziness_values = compute_laziness_values(answer_embedding, evidence["neighbour_reference_answers"], api_key)
    single_similarity = cosine_similarity(answer_embedding, single_embedding)
    single_laziness = compute_single_laziness(answer_embedding, evidence["neighbour_single_answers"], api_key)

    # Here we build the "no penalty" and "equal weights" inputs used by the comparison rows.
    number_of_references = len(answer_similarities)
    equal_weights = [1 / number_of_references] * number_of_references
    no_laziness = [0.0] * number_of_references
    weights = evidence["expertise_weights"]

    return {
        "single_no_penalty": g_star(single_similarity),
        "single_with_penalty": g_star(single_similarity) - f_star(g_star(single_laziness)),
        "multi_no_penalty": compute_fewl_score(answer_similarities, no_laziness, equal_weights),
        "fewl": compute_fewl_score(answer_similarities, laziness_values, weights),
        "fewl_section_3_1": compute_section_3_1_score(answer_similarities, laziness_values, weights),
    }


def main():
    """Demo: run FEWL on one made-up question and print the scores of a right and a wrong answer."""
    # Here we load the API key; step over this line to check it was found.
    api_key = load_api_key(ENV_FILE)

    question = "What is the capital city of Australia?"
    question_pool = [
        "Which river flows through Cairo?",
        "What is the largest desert in Africa?",
        "Who painted the ceiling of the Sistine Chapel?",
        "What currency is used in Japan?",
        "Which ocean lies between Africa and Australia?",
        "Who wrote the novel Pride and Prejudice?",
        "What is the tallest mountain in North America?",
        "Which planet is known as the Red Planet?",
        "What language is mainly spoken in Brazil?",
        "Which country has the most islands in the world?",
        "Who was the first person to walk on the Moon?",
        "What is the longest river in Europe?",
    ]
    pool_embeddings = get_embeddings(question_pool, api_key)

    # Here we find the neighbours first; this made-up question is not in the pool, so we embed it on its own.
    question_embedding = get_embeddings([question], api_key)[0]
    neighbour_questions = find_neighbour_questions(question_embedding, question_pool, pool_embeddings)

    evidence = gather_evidence(question, neighbour_questions, api_key)
    print(f"Question: {question}")
    print(f"Reference answers: {evidence['reference_answers']}")
    print(f"Expertise weights: {[round(weight, 3) for weight in evidence['expertise_weights']]}")
    print(f"Neighbour questions: {evidence['neighbour_questions'][:3]} ...\n")

    for answer in ["Canberra is the capital of Australia.", "Sydney is the capital of Australia."]:
        scores = score_answer(answer, evidence, api_key)
        rounded_scores = {name: round(score, 4) for name, score in scores.items()}
        print(f"{answer}\n  {rounded_scores}")


if __name__ == "__main__":
    main()
