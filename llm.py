"""Send questions to an OpenAI model and get its answers back.

Before the first run, create a file called .env next to this one, holding your key on one line:

    OPENAI_API_KEY=sk-...

Then run it with Debug in VS Code or `python3 llm.py`. The demo asks the model a question and prints the answer.

Every reply is saved in APICache/, so sending the exact same request again costs nothing and returns
the same reply. Delete that folder if you ever want fresh replies.
"""

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Here we keep every setting in one place, so it is easy to see and change.
ENV_FILE = Path(__file__).parent / ".env"
CACHE_FOLDER = Path(__file__).parent / "APICache"
API_URL = "https://us.api.openai.com/v1/chat/completions"
MODEL_NAME = "gpt-4.1-mini"
DEFAULT_MAX_ANSWER_TOKENS = 1000
REQUEST_TIMEOUT_SECONDS = 180       # give up on a reply that takes longer than this
MAX_REQUEST_ATTEMPTS = 5            # tries per request when OpenAI is busy or the connection drops
RETRY_WAIT_SECONDS = 10             # wait before the 2nd try; it grows by this much on each later try
TOO_MANY_REQUESTS_CODE = 429        # OpenAI is asking us to slow down


def load_api_key(env_file):
    """Read the OpenAI API key from the .env file."""
    if not env_file.exists():
        raise FileNotFoundError(f"Create {env_file} containing the line OPENAI_API_KEY=sk-...")

    with open(env_file, encoding="utf-8") as file:
        for line in file:
            # Here we look for the line that holds the key and keep only the part after the "=".
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()

    raise ValueError(f"No OPENAI_API_KEY line found in {env_file}")


def send_request_once(url, request_body, api_key):
    """Send one request to the OpenAI API and return its reply as a dictionary."""
    request = urllib.request.Request(
        url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )

    # Here we send the request over the internet and wait for OpenAI's reply.
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.load(response)


def read_error_text(error):
    """Turn a failed request into readable text, using OpenAI's own explanation when there is one."""
    if isinstance(error, urllib.error.HTTPError):
        return error.read().decode("utf-8")
    return str(error)


def is_worth_retrying(error, error_text):
    """Say whether a failed request may work if sent again: OpenAI was busy, or the connection dropped."""
    if isinstance(error, urllib.error.HTTPError):
        # Here we give up straight away when the account is out of credit, which also comes back as 429.
        if "insufficient_quota" in error_text:
            return False

        # Here every 5xx code is a server-side hiccup worth another try, including Cloudflare's 520-524,
        # which OpenAI's servers sit behind.
        is_server_error = error.code >= 500
        return error.code == TOO_MANY_REQUESTS_CODE or is_server_error

    # Here any other error is a dropped connection or a timeout, which is worth another try.
    return True


def send_request(url, request_body, api_key):
    """Send one request, waiting and trying again when OpenAI is busy or the connection drops."""
    for attempt_number in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            return send_request_once(url, request_body, api_key)
        except OSError as error:
            # Here OSError covers every network failure: HTTP errors, dropped connections and timeouts.
            error_text = read_error_text(error)
            if attempt_number == MAX_REQUEST_ATTEMPTS or not is_worth_retrying(error, error_text):
                # Here we show OpenAI's own explanation (e.g. wrong key, unknown model) instead of just an error number.
                raise RuntimeError(error_text) from error

            # Here we wait a little longer after each failed try, to give OpenAI time to catch up.
            wait_seconds = RETRY_WAIT_SECONDS * attempt_number
            print(f"Request failed (try {attempt_number}/{MAX_REQUEST_ATTEMPTS}), trying again in {wait_seconds} s: "
                  f"{error_text[:100]}")
            time.sleep(wait_seconds)


def build_cache_path(url, request_body, cache_folder):
    """Turn a request into a file name, so the same request always maps to the same cache file."""
    request_text = json.dumps({"url": url, "body": request_body}, sort_keys=True)

    # Here we turn the request text into a short fingerprint that is safe to use as a file name.
    request_fingerprint = hashlib.sha256(request_text.encode("utf-8")).hexdigest()
    return cache_folder / f"{request_fingerprint}.json"


def save_reply_safely(cache_path, response_body):
    """Save a reply under a temporary name, then rename it, so no one ever reads a half-written cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Here each thread writes its own temporary file, so two threads saving at once do not mix their text.
    temporary_path = cache_path.parent / f"{cache_path.stem}_{threading.get_ident()}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(response_body, file)

    # Here the rename happens in one go, so the cache file is either complete or not there at all.
    temporary_path.replace(cache_path)


def send_request_with_cache(url, request_body, api_key, cache_folder):
    """Return the saved reply if this exact request was sent before; otherwise send it and save the reply."""
    cache_path = build_cache_path(url, request_body, cache_folder)

    # Here we reuse the saved reply, which makes re-runs free and gives the same result every time.
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as file:
            return json.load(file)

    response_body = send_request(url, request_body, api_key)
    save_reply_safely(cache_path, response_body)
    return response_body


def ask_model_for_answers(prompt, model_name, api_key, number_of_answers, temperature, max_answer_tokens):
    """Send one prompt and return several separate answers to it, all from a single API call."""
    # Here "n" asks for several answers at once, "temperature" sets how much they vary (0 = always the same),
    # and "max_completion_tokens" stops an answer that runs too long.
    request_body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "n": number_of_answers,
        "temperature": temperature,
        "max_completion_tokens": max_answer_tokens,
    }
    response_body = send_request_with_cache(API_URL, request_body, api_key, CACHE_FOLDER)

    # Here we pull the answer texts out of the reply, which also carries extra details such as token counts.
    answers = []
    for choice in response_body["choices"]:
        answers.append(choice["message"]["content"].strip())

        # Here we warn when the limit cut an answer off, so it is visible how often that happens.
        if choice["finish_reason"] == "length":
            print(f"Warning: an answer was cut off at {max_answer_tokens} tokens. Prompt: {prompt[:80]!r}")
    return answers


def ask_model(question, model_name, api_key):
    """Send one question to the model and return its answer as text."""
    answers = ask_model_for_answers(
        question, model_name, api_key, number_of_answers=1, temperature=1.0, max_answer_tokens=DEFAULT_MAX_ANSWER_TOKENS
    )
    return answers[0]


def ask_many_questions(questions, model_name, api_key):
    """Ask each question in turn and return the answers in the same order."""
    answers = []
    for question in questions:
        answers.append(ask_model(question, model_name, api_key))
    return answers


def main():
    """Demo: ask the model a short question and print the answer."""
    # Here we load the API key; step over this line to check it was found.
    api_key = load_api_key(ENV_FILE)

    questions = [
        "This is a test. Respond with \"I can hear you.\""
    ]
    answers = ask_many_questions(questions, MODEL_NAME, api_key)

    for question, answer in zip(questions, answers):
        print(f"Q: {question}\nA: {answer}\n")


if __name__ == "__main__":
    main()
