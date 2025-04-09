SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

MULTIPLICATION4x4_PROMPT = """
Respond in the following format:
<reasoning>
<step>...</step>
...
<step>...</step>
</reasoning>
<answer>...</answer>

Step by step, multiply the numbers. For each place value of the multiplier, provide the intermediate multiplication result in the form of <step>multiplicand * place value = result</step>. Sum all the intermediate results and provide the final result in the <answer>...</answer>.
"""

MULTIPLICATION4x4_PROMPT_EXTENDED = """
Respond in the following format:
<reasoning>
<step>...</step>
...
<step>...</step>
</reasoning>
<answer>...</answer>

Step by step, multiply the numbers. For each place value of the multiplier, provide the intermediate multiplication result in the form of <step>multiplicand * place value = result</step>. Sum all the intermediate results and provide the final result in the <answer>...</answer>.
"""


def gsm8k_reasoning_prompt(example):
    final_answer = example['answer'].split('####')[1].strip().replace(',', '')
    # Sanity check: final answer is a number
    float(final_answer)
    return {'reasoning_prompt': f'{SYSTEM_PROMPT.strip()}\n{example["question"].strip()}', 'final_answer': final_answer}


def multiplication4x4_reasoning_prompt(example):
    multiplier, multiplicand = example['task'][::-1].replace(' ', '').split('*')
    answer = example['labels'][::-1].replace(' ', '')
    assert int(multiplicand) * int(multiplier) == int(answer)
    gt_step_strings = []
    for place, digit in enumerate(multiplier[::-1]):
        place_value = f'{digit}{"0" * place}'
        step_answer = int(multiplicand) * int(place_value)
        gt_step_strings.append(f'{multiplicand} * {place_value} = {step_answer}')
    gt_steps = '\n'.join(gt_step_strings)
    return {'reasoning_prompt': f'{MULTIPLICATION4x4_PROMPT.strip()}\n{multiplicand} * {multiplier}', 'final_answer': f'{gt_steps}\n{answer}'}