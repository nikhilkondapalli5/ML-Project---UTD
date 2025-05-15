from typing import List, Callable, Union, Optional
from collections import deque
from dataclasses import dataclass, field
from IPython.display import display, HTML
import openai  # pip install openai

# ----------------- TreeNode -----------------
@dataclass
class TreeNode:
    state: str
    thought: str
    value: float = 0.0
    children: List['TreeNode'] = field(default_factory=list)

# ----------------- Prompt Functions -----------------
def get_thought_gen_prompt(input_seq: str, state: str) -> str:
    return f"{input_seq}\n{state}\nThink about how the story could continue. Write a creative continuation. Passage:"

def get_state_eval_prompt(input_seq: str, states: List[str]) -> str:
    formatted_states = "\n\n".join(f"Option {i+1}:\n{state}" for i, state in enumerate(states))
    return f"""Evaluate the following story continuations and vote for the most creative and engaging one.

{formatted_states}

Which option is the best? Respond like: "Option 2"
"""

def heuristic_calculator(states: List[str], votes: List[str]) -> List[float]:
    scores = [0] * len(states)
    for vote in votes:
        for i in range(len(states)):
            if f"Option {i+1}" in vote:
                scores[i] += 1
    return scores

# ----------------- Tree of Thoughts -----------------
class TreeOfThoughts:
    def __init__(
        self,
        client: Union[openai.OpenAI, any],
        model: str,
        input_seq: str,
        get_thought_gen_prompt: Callable,
        get_state_eval_prompt: Callable,
        heuristic_calculator: Callable
    ):
        self.client = client
        self.model = model
        self.input_seq = input_seq
        self.root = TreeNode(state='', thought='')
        self.n_steps = 2
        self.thought_gen_strategy = 'sample'
        self.get_thought_gen_prompt = get_thought_gen_prompt
        self.n_candidates = 5
        self.stop_string = 'Passage:'
        self.state_eval_strategy = 'vote'
        self.get_state_eval_prompt = get_state_eval_prompt
        self.n_evals = 5
        self.heuristic_calculator = heuristic_calculator
        self.breadth_limit = 1

    def chat_completions(self, prompt: str, temperature=0.7, max_tokens=1000, n=1, stop=None, **kwargs) -> List[str]:
        outputs = []
        messages = [{'role': "user", 'content': prompt}]
        if isinstance(self.client, openai.OpenAI):  # OpenAI client
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                stop=stop,
                **kwargs
            )
            outputs = [choice.message.content for choice in response.choices]
        else:  # HuggingFace InferenceClient
            for _ in range(n):
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    **kwargs
                )
                outputs.append(response.choices[0].message.content)
        return outputs

    def thought_generator(self, state: str, stop_string: Optional[List[str]] = None) -> List[str]:
        prompt = self.get_thought_gen_prompt(self.input_seq, state)
        return self.chat_completions(prompt, n=self.n_candidates, stop=stop_string)

    def state_evaluator(self, states: List[str]) -> List[float]:
        prompt = self.get_state_eval_prompt(self.input_seq, states)
        state_evals = self.chat_completions(prompt, n=self.n_evals)
        return self.heuristic_calculator(states, state_evals)

    def bfs(self, verbose: bool = True) -> str:
        queue = deque()
        queue.append(self.root)

        for step in range(1, self.n_steps + 1):
            if verbose:
                print(f"Step {step}:\n---")
            for _ in range(len(queue)):
                node = queue.popleft()
                thoughts = self.thought_generator(node.state, [self.stop_string] if step == 1 else None)
                updated_states = [t if node.state == '' else node.state + '\n' + t for t in thoughts]
                for t, s in zip(thoughts, updated_states):
                    child = TreeNode(state=s, thought=t)
                    node.children.append(child)
                    queue.append(child)

            states = [node.state for node in queue]
            values = self.state_evaluator(states)
            for i in range(len(queue)):
                queue[i].value = values[i]

            sorted_nodes = sorted(queue, key=lambda node: node.value, reverse=True)
            queue = deque(sorted_nodes[:1] if step == self.n_steps else sorted_nodes[:self.breadth_limit])

        return queue[0].thought

    def generate_html_tree(self, node: TreeNode) -> str:
        if node is None:
            return ""
        html = f"""<div class='node'>
<p><b>State:</b><br>{node.state}</p><hr>
<p><b>Thought:</b><br>{node.thought}</p><hr>
<p><b>Value:</b><br>{node.value}</p>"""
        for child in node.children:
            html += f"""<div class='child'>{self.generate_html_tree(child)}</div>"""
        return html + "</div>"

    def render_html_tree(self):
        html_tree = self.generate_html_tree(self.root)
        wrapped_html = f"""<!DOCTYPE html>
<html><head>
<style>
.node {{ display:inline-block; border:1px solid blue; padding:10px; margin:5px; text-align:left; background:#f9f9f9 }}
.child {{ display:flex; }}
</style></head><body>{html_tree}</body></html>"""
        display(HTML(wrapped_html))

# ----------------- Example Usage -----------------
if __name__ == "__main__":
    client = openai.OpenAI(api_key="your_openai_api_key")  # Replace with your OpenAI API key
    prompt = "Write a story about a time-traveling violinist who visits ancient civilizations."

    tot = TreeOfThoughts(
        client=client,
        model="gpt-4o-mini",  # or "gpt-4" or any HuggingFace-compatible model
        input_seq=prompt,
        get_thought_gen_prompt=get_thought_gen_prompt,
        get_state_eval_prompt=get_state_eval_prompt,
        heuristic_calculator=heuristic_calculator
    )

    best_thought = tot.bfs(verbose=True)
    print("\n Final Thought Chosen:\n", best_thought)
