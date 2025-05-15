import os
import logging
import json
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from sympy import sympify
from multimodal_retriever import MultimodalRetriever

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Calculator tool schema
calculator_tool_schema = {
    "type": "function",
    "function": {
        "name": "calculate_expression",
        "description": "Evaluates math expressions like growth ((new-old)/old*100), USD<->INR.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math formula string."}
            },
            "required": ["expression"],
            "additionalProperties": False
        }
    }
}

class RAGGenerator:
    def __init__(self, temperature: float = 0.1, top_p: float = 1.0, seed: Optional[int] = None):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.retriever = MultimodalRetriever()
        self.model = "gpt-4o-mini"
        self.conversation_history = []
        self.max_history_length = 10

        # Hyperparameters
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

    def _calculate_expression(self, expression: str) -> str:
        try:
            expr_cleaned = (expression.replace(',', '').replace(' ', '').replace('$', '').replace('%', '')
                            .replace('₹', '').upper().replace('M', '*1e6').replace('K', '*1e3'))
            result_float = float(sympify(expr_cleaned).evalf())
            return f"{result_float:.2f}"
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return f"Calculation Error: {str(e)}"

    def _format_context(self, retrieval_results: Dict) -> str:
        context = "RELEVANT DOCUMENT SECTIONS:\n\n"
        for i, result in enumerate(retrieval_results['text_results'], 1):
            context += f"TEXT SECTION {i} (Page {result['metadata'].get('page_number', 'unknown')}):\n"
            context += f"{result['text'].strip()}\n\n"
        if retrieval_results['image_results']:
            context += "RELEVANT IMAGES:\n"
            for i, result in enumerate(retrieval_results['image_results'], 1):
                context += f"IMAGE {i}: From page {result['page_number']}, path: {result['image_path']}\n"
        return context

    def _build_prompt(self, query: str, context: str) -> tuple:
        system_prompt = """You are an AI assistant for corporate document QA.
Use ONLY the context below. Use the 'calculate_expression' tool for math.
If context is insufficient, say so. Cite page numbers.
"""
        user_prompt_content = f"{context}\n\nQUESTION: {query}\n\nANSWER:"
        return system_prompt, user_prompt_content

    def _prepare_messages_with_history(self, system_prompt: str, user_prompt: str) -> List[Dict]:
        messages = [{"role": "system", "content": system_prompt}]
        for entry in self.conversation_history:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _format_sources(self, retrieval_results: Dict) -> List[Dict]:
        sources = []
        for result in retrieval_results.get('text_results', []):
            sources.append({
                'type': 'text', 'page': result.get('metadata', {}).get('page_number'),
                'content_preview': result.get('text', '')[:100] + "..."
            })
        for result in retrieval_results.get('image_results', []):
            sources.append({'type': 'image', 'page': result.get('page_number'), 'path': result.get('image_path')})
        return sources

    def generate_answer(self, query: str, n_text_results: int = 5, n_image_results: int = 3, tools: List[Dict] = None) -> Dict:
        final_answer = "Sorry, an error occurred."
        sources = []
        try:
            retrieval_results = self.retriever.hybrid_query(query, n_text_results, n_image_results)
            sources = self._format_sources(retrieval_results)
            context = self._format_context(retrieval_results)
            system_prompt, user_prompt = self._build_prompt(query, context)
            messages = self._prepare_messages_with_history(system_prompt, user_prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or [],
                tool_choice="auto" if tools else None,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed
            )
            response_message = response.choices[0].message
            if response_message.tool_calls:
                messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name != "calculate_expression": continue
                    args = json.loads(tool_call.function.arguments)
                    expr = args.get("expression")
                    if expr:
                        tool_result = self._calculate_expression(expr)
                        messages.append({"tool_call_id": tool_call.id, "role": "tool",
                                         "name": tool_call.function.name, "content": tool_result})
                second_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=self.seed
                )
                final_answer = second_response.choices[0].message.content
            else:
                final_answer = response_message.content
            self.conversation_history.append({"user": query, "assistant": final_answer})
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history.pop(0)
            return {'query': query, 'answer': final_answer, 'sources': sources}
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return {'query': query, 'answer': f"Error: {str(e)}", 'sources': sources}

    def generate_cot_answer(self, query: str) -> Dict:
        return self.generate_answer("Let's think step-by-step. " + query)

    def generate_react_answer(self, query: str) -> Dict:
        return self.generate_answer("Use ReAct strategy: think, then act (tool), then continue. " + query, tools=[calculator_tool_schema])

    def generate_tot_answer(self, query: str) -> Dict:
        paths = [f"Path {i+1}: {query}" for i in range(3)]
        results = [self.generate_answer(p) for p in paths]
        eval_prompt = "Evaluate the following responses and score them from 1 to 10. Then choose the best.\n\n"
        for i, res in enumerate(results, 1):
            eval_prompt += f"--- Response {i} ---\n{res['answer']}\n\n"
        eval_prompt += "Output format:\nScore 1: <number>\nScore 2: <number>\nScore 3: <number>\nBest: <number>"
        eval = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an impartial evaluator."},
                {"role": "user", "content": eval_prompt},
            ],
            temperature=0.5,
            top_p=1.0,
            seed=self.seed
        )
        eval_text = eval.choices[0].message.content
        match = re.search(r"Best:\s*(\d)", eval_text)
        best_index = int(match.group(1)) - 1 if match else max(range(3), key=lambda i: len(results[i]['answer']))
        return {
            "query": query,
            "answer": results[best_index]['answer'],
            "paths": [r['answer'] for r in results],
            "scores": eval_text,
            "sources": results[best_index]['sources']
        }

    def reset_conversation(self):
        self.conversation_history = []
        logger.info("Conversation history reset.")

if __name__ == "__main__":
    generator = RAGGenerator(temperature=1.0, top_p=0.95, seed=42)
    q = "What are key insights from the report? Include any calculations."

    print("\n--- CoT ---")
    print(generator.generate_cot_answer(q)['answer'])

    print("\n--- ReAct ---")
    print(generator.generate_react_answer(q)['answer'])

    print("\n--- ToT ---")
    tot = generator.generate_tot_answer(q)
    print(tot['answer'])
    print("\nScores:\n", tot['scores'])
    print("\nAll Paths:")
    for i, p in enumerate(tot['paths']):
        print(f"Path {i+1}: {p}\n")

    print("\n--- Normal ---")
    print(generator.generate_answer(q)['answer'])