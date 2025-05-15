# Tree-of-Thoughts Story Generator

A Python implementation of the **Tree-of-Thoughts** framework for creative story continuation.  
Uses OpenAI’s chat API (or a HuggingFace Inference client) to **generate**, **evaluate**, and **prune** multiple candidate continuations, then selects the most engaging branch via an LLM-based voting heuristic. Includes optional HTML visualization of the search tree.

---

## 🔍 Features

- **Multi-step exploration**  
  Breadth-first search over several “thought” continuations at each step.  
- **Candidate generation**  
  Sample N continuations per state using a user-configurable LLM and temperature.  
- **Voting-based evaluation**  
  Ask the LLM to vote on its favorite continuation among a set of candidates.  
- **Heuristic scoring**  
  Tally “Option X” votes into numerical values to rank branches.  
- **Branch pruning**  
  Keep only the top-K highest-scoring states at each step to control search complexity.  
- **HTML visualization**  
  Render the entire exploration tree in Jupyter with inline HTML/CSS for inspection.

---

## 📋 Requirements

- Python 3.8+  
- `openai` (or your HuggingFace Inference client)  
- `dataclasses` (built-in on 3.7+, backport for older)  
- `IPython` for HTML display  
- A valid OpenAI API key (or HF token)

---

## ⚙️ Installation

1. **Clone this repo**  
   ```bash
   git clone https://github.com/your-org/tree-of-thoughts.git
   cd tree-of-thoughts

python3 -m venv venv
source venv/bin/activate

pip install openai ipython

export OPENAI_API_KEY="sk-..."

Usage:
python tree_of_thoughts.py

Configuration Options:

n_steps: number of exploration rounds
n_candidates: continuations sampled per node
breadth_limit: how many top branches to keep per round
temperature, max_tokens: LLM sampling parameters
n_evals: how many votes to collect for evaluation

All parameters can be adjusted on the TreeOfThoughts instance after initialization.
