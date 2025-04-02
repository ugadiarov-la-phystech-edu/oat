SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def gsm8k_reasoning_prompt(example):
    final_answer = example['answer'].split('####')[1].strip().replace(',', '')
    # Sanity check: final answer is a number
    float(final_answer)
    return {'reasoning_prompt': f'{SYSTEM_PROMPT.strip()}\n{example["question"].strip()}', 'final_answer': final_answer}