# HalluTrace

HalluTrace tests ways to **detect hallucinations**: answers from a language model that sound right but are false.

The first method here is **FEWL**, from Wei et al., *"Measuring and Reducing LLM Hallucination without Gold-Standard Answers"* ([arXiv 2402.10412](https://arxiv.org/abs/2402.10412)). FEWL scores an answer without knowing the right answer. It asks a reference model (here OpenAI's `gpt-4.1-mini`) the same question several times. Then it checks:

- how well the answer agrees with the model's answers;
- whether the answer is "lazy", meaning it would fit similar questions just as well.

We test it on the [HaluEval](https://github.com/RUCAIBox/HaluEval) benchmark. Every HaluEval question comes with one right answer and one hallucinated answer. A good method gives the right answer the higher score.

---

## What you need

- **A computer with Python 3.9 or newer.** No extra packages are needed, because everything uses Python's standard library.
- **An OpenAI API key** with some credit on the account. You can create one at <https://platform.openai.com/api-keys>.
- **About 500 MB of free disk space**. The dataset is about 120 MB, and saved API replies grow with every run.
- *(Optional)* **[VS Code](https://code.visualstudio.com/)** with the Python extension, so you can step through the code with breakpoints.

---

## Getting started

Do these steps in order. Each one tells you what you should see.

### 1. Check that Python is installed

Open a terminal (on a Mac: the *Terminal* app) and type:

```bash
python3 --version
```

You should see something like `Python 3.9.6` or higher.

- If you see "command not found", install Python from <https://www.python.org/downloads/> and try again.
- **Windows:** type `python` instead of `python3` here and in every later step.

### 2. Download the code

```bash
git clone https://github.com/shoimya/HalluTrace.git
cd HalluTrace
```

From now on, run every command from inside this `HalluTrace` folder.

### 3. Add your OpenAI API key

Create a file called `.env` (the dot at the start matters) directly inside the `HalluTrace` folder. It should hold one line:

```
OPENAI_API_KEY=sk-your-key-here
```

The `.env` file is listed in `.gitignore`, so git will never upload it. **Never paste your key into any other file.**

> **About the address the code calls:** the code sends requests to `https://us.api.openai.com/...`. That address is for OpenAI projects set to **US data residency**. If your key comes from an ordinary project and step 4 fails with an authentication or project error, change the two addresses at the top of these files:
> - `API_URL` in [llm.py](llm.py) → `https://api.openai.com/v1/chat/completions`
> - `EMBEDDINGS_URL` in [embeddings.py](embeddings.py) → `https://api.openai.com/v1/embeddings`

### 4. Check that the key works

```bash
python3 llm.py
```

You should see:

```
Q: This is a test. Respond with "I can hear you."
A: I can hear you.
```

This costs a fraction of a cent. If you get an error instead, see [When something goes wrong](#when-something-goes-wrong).

### 5. Download the HaluEval dataset

```bash
python3 DatasetRetrievers/HaluEvaldataset.py
```

This downloads the four HaluEval tasks from GitHub, which takes under a minute. Then it splits each task into small, fixed pieces. You should see lines like:

```
qa: downloading https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json
qa: 10000 examples -> 10 pilot, 9990 test
qa: 100 examples saved to sample_100.jsonl
...
Done. Files are in .../HaluEvalData
```

The files land in `HaluEvalData/`:

```
HaluEvalData/raw/qa_data.json       the original download, never changed
HaluEvalData/qa/pilot.jsonl         10 questions, for a quick, cheap first try
HaluEvalData/qa/sample_100.jsonl    100 random questions (the pilot's 10 are the first 10)
HaluEvalData/qa/test.jsonl          everything else, kept for the real evaluation
```

The same layout is made for `dialogue`, `summarization` and `general`. The shuffle uses a fixed seed, so everyone gets exactly the same files.

### 6. Point the run at the pilot set

Open [run.py](run.py) and find this line near the top:

```python
INPUT_FILE = Path(__file__).parent / "HaluEvalData" / "qa" / "sample_1000.jsonl"
```

Change `sample_1000.jsonl` to `pilot.jsonl` and save. You have to do this because a fresh download has no `sample_1000.jsonl` (see [Running on more questions](#running-on-more-questions)).

### 7. Run FEWL on the 10 pilot questions

```bash
python3 run.py
```

Or open `run.py` in VS Code and press **Run → Start Debugging** (F5).

What happens:

1. **`Embedded 10000 pool questions`**: every question in the full QA file is turned into numbers, so similar "neighbour" questions can be found. The first time, this is about 100 requests to OpenAI. After that it is read from disk.
2. **`Found neighbours for 10 questions`**: each pilot question gets its 10 most similar questions.
3. **One line per question**, for example:
   ```
   [3/10] fewl right=-0.0601 hallucinated=-0.1046 | Rob Jenkins is an Australian actor, ...
   ```
   The right answer should score **higher** than the hallucinated one. Each line is followed by a long `details:` line with all the numbers, which you can skip for now.
4. **A results table at the end**:
   ```
   Accuracy on 10 examples (right answer scored higher than hallucinated):
     single_no_penalty     ...%   paper (HaluEval QA): 94.33%
     single_with_penalty   ...%   paper (HaluEval QA): 97.89%
     multi_no_penalty      ...%   paper (HaluEval QA): 95.52%
     fewl                  ...%   paper (HaluEval QA): 98.15%
     fewl_section_3_1      ...%   paper (HaluEval QA): not reported
   Log saved to runs/fewl/<date>/<date>_<time>_qa_pilot.log
   ```

The pilot takes a few minutes and costs a few cents. With only 10 questions the percentages jump around a lot, so don't read much into them yet.

### 8. Read the results

Every row answers one question: **out of all the questions, how often did the right answer score higher than the hallucinated one?** 50% is no better than a coin flip.

| Row | How the answer is scored |
|---|---|
| `single_no_penalty` | Similarity to **one** model answer (temperature 0). |
| `single_with_penalty` | The same, minus its similarity to the model's answers for the neighbour questions (the "lazy" penalty). |
| `multi_no_penalty` | Average similarity to **five** model answers. |
| `fewl` | The paper's Equation 1: five answers, each weighted by how expert it looks, minus the lazy penalty. |
| `fewl_section_3_1` | Same ingredients, with the weight applied the way the paper's Section 3.1 describes it. |

The full run, with every message and time, is saved in the log file named on the last line.

---

## Running on more questions

- **100 questions:** set `INPUT_FILE` in `run.py` to `sample_100.jsonl`.
- **1000 questions:** open `DatasetRetrievers/HaluEvaldataset.py` and set `SAMPLE_SIZE = 1000`. Run it again (it skips the download because the files are already on disk), then set `INPUT_FILE` to `sample_1000.jsonl`. As a guide, 1000 questions took about 40 minutes.
- **Another task:** change `"qa"` in `INPUT_FILE` to `"dialogue"` or `"summarization"`. The paper only reports FEWL on QA.

Every setting sits as an `UPPER_CASE` constant at the top of its file, with a comment saying what it does. Settings the paper doesn't give are marked `NOT IN PAPER`.

## Saved replies: re-runs are free

Every reply from OpenAI is saved as a file in `APICache/`. If the code ever sends the **exact same request** again, it reads the saved reply instead of asking OpenAI. This means:

- running the same thing twice costs nothing the second time and gives the same result;
- a run that crashes halfway can simply be started again, and it skips what is already done.

Delete the `APICache/` folder if you want fresh answers from the model.

---

## What is in the folder

```
HalluTrace/
├── README.md                 this file
├── run.py                    START HERE: runs FEWL on a HaluEval file and prints the accuracy table
├── fewl.py                   the FEWL method itself (paper's Algorithm 1 and Equation 1)
├── llm.py                    sends questions to OpenAI, retries when it is busy, saves replies in APICache/
├── embeddings.py             turns text into numbers and measures how similar two texts are
├── DatasetRetrievers/
│   ├── HaluEvaldataset.py    downloads HaluEval and makes the pilot / sample / test files
│   └── CHALEdataset.py       placeholder for the CHALE dataset (not written yet)
├── Notebook/
│   └── run_fewl_on_colab.ipynb   a notebook for running on Google Colab instead of your computer
├── Resources/                the papers this work is based on, and progress slides
│
│   Made by the code (not in git):
├── .env                      your API key (you create this in step 3)
├── HaluEvalData/             the dataset (step 5)
├── APICache/                 saved OpenAI replies
└── runs/                     one log file per run: runs/<method>/<date>/<date>_<time>_<task>_<split>.log
```

**Every file can be run on its own** with `python3 <file>` or the Debug button. Each one ends with a small demo, so you can try one piece at a time:

| File | Its demo |
|---|---|
| `llm.py` | asks the model one test question |
| `embeddings.py` | prints how similar two short texts are |
| `fewl.py` | runs FEWL on one made-up question with a right and a wrong answer |
| `DatasetRetrievers/HaluEvaldataset.py` | downloads and splits the dataset |
| `run.py` | the full run described above |

To follow the code step by step, open a file in VS Code, click left of a line number to set a red breakpoint dot, and press F5. The run pauses there so you can inspect each variable.

---

## When something goes wrong

| What you see | What it means and what to do |
|---|---|
| `FileNotFoundError: Create .../.env containing the line OPENAI_API_KEY=sk-...` | The `.env` file is missing or in the wrong folder. It must sit directly inside `HalluTrace/`, next to `run.py` (step 3). |
| `No OPENAI_API_KEY line found` | The `.env` file exists but the line is misspelt. It must start exactly with `OPENAI_API_KEY=`, with no spaces. |
| An error mentioning `invalid_api_key`, `project` or `401` | The key is wrong, or it isn't a US data-residency key. See the note in step 3. |
| An error mentioning `insufficient_quota` | Your OpenAI account has no credit left. Add some on the OpenAI billing page. |
| `Request failed (try 1/5), trying again in 10 s` | OpenAI is busy for a moment. The code waits and tries again by itself, so you don't need to do anything. |
| `FileNotFoundError: ... sample_1000.jsonl` | That file isn't made by default. Do step 6, or see [Running on more questions](#running-on-more-questions). |
| `FileNotFoundError: ... HaluEvalData/...` | The dataset hasn't been downloaded yet. Do step 5. |
| `python3: command not found` | Python isn't installed, or on Windows it is called `python` (step 1). |

---

## Where the project stands

- FEWL runs end to end on HaluEval QA.
- On 1000 QA questions it scores well below the paper (for example `fewl` 53.4% against 98.15%).
- The main suspect is the similarity measure. Right now answers are compared with **cosine similarity of OpenAI embeddings**, which measures whether two texts *look* alike. The paper used a model from **Vectara**, most likely [HHEM](https://huggingface.co/vectara/hallucination_evaluation_model), which judges whether two texts state *the same facts*.
- Switching to HHEM is the next piece of work. It will need a few extra packages, and this README will say how to install them when that lands.
