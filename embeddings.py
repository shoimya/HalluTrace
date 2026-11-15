"""Measure how similar two pieces of text are in meaning, using OpenAI's embedding model.

An embedding is a list of numbers that captures what a text means: texts with similar meanings get
similar lists. Cosine similarity turns two embeddings into one number, where 1 means the same meaning
and values near 0 mean unrelated.

Run it with Debug in VS Code or `python3 embeddings.py`. The demo compares a few sentences and prints
how similar each pair is. Replies are cached in APICache/, like in llm.py.
"""

import math

from llm import CACHE_FOLDER, ENV_FILE, load_api_key, send_request_with_cache

# Here we keep every setting in one place, so it is easy to see and change.
EMBEDDINGS_URL = "https://us.api.openai.com/v1/embeddings"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_SIZE = 256
TEXTS_PER_REQUEST = 100


def get_embeddings(texts, api_key):
    """Turn each text into its embedding, sending the texts to OpenAI in batches."""
    embeddings = []
    for batch_start in range(0, len(texts), TEXTS_PER_REQUEST):
        batch_of_texts = texts[batch_start:batch_start + TEXTS_PER_REQUEST]

        # Here "dimensions" asks for shorter embeddings, which keeps the cache small and comparisons fast.
        request_body = {"model": EMBEDDING_MODEL_NAME, "input": batch_of_texts, "dimensions": EMBEDDING_SIZE}
        response_body = send_request_with_cache(EMBEDDINGS_URL, request_body, api_key, CACHE_FOLDER)

        for item in response_body["data"]:
            embeddings.append(item["embedding"])
    return embeddings


def cosine_similarity(first_embedding, second_embedding):
    """Return how similar two embeddings are: 1 means the same meaning, around 0 means unrelated."""
    dot_product = 0.0
    for first_value, second_value in zip(first_embedding, second_embedding):
        dot_product += first_value * second_value

    # Here we divide by both lengths, so only the direction of the embeddings counts, not their size.
    first_length = math.sqrt(sum(value * value for value in first_embedding))
    second_length = math.sqrt(sum(value * value for value in second_embedding))
    return dot_product / (first_length * second_length)


def main():
    """Demo: embed three sentences and print how similar each pair is."""
    # Here we load the API key; step over this line to check it was found.
    api_key = load_api_key(ENV_FILE)

    # sentences = [
    #     "Antonio Vivaldi",
    #     "Antonio Lucio Vivaldi"
    # ]
    sentences = [
        "J.R.R. Tolkien",
        "J. R. R. Tolkien"
    ]
    embeddings = get_embeddings(sentences, api_key)

    for first_index in range(len(sentences)):
        for second_index in range(first_index + 1, len(sentences)):
            similarity = cosine_similarity(embeddings[first_index], embeddings[second_index])
            print(f"{similarity:.3f}  {sentences[first_index]!r} vs {sentences[second_index]!r}")


if __name__ == "__main__":
    main()
