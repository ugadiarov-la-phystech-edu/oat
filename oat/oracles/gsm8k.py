# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Optional, Tuple, Collection

import regex as re
import torch

from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric

FIND_NUMBERS_REGEX = re.compile(
    r"(?:[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+e[-+]?\d+|[+-]?\d+)"
)


def contains_any(string: str, substrings: Collection[str]):
    for s in substrings:
        if s in string:
            return True

    return False


class GSM8KOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the GSM8K task."""

    def __init__(self, use_original_format: bool = False, **_) -> None:
        super().__init__()
        self.use_original_format = use_original_format

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        del inputs, batch_size
        predicted_answers = []
        rewards = []

        for resp, ref in zip(responses, references):
            answer_candidate = self._extract_predicted_answer_from_text(resp)
            predicted_answers.append(answer_candidate)
            grading_res = self._grade_answer(answer_candidate, ref)
            rewards.append(float(grading_res))

        return torch.tensor(rewards), {"predicted_answers": predicted_answers}

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info

    def _extract_predicted_answer_from_text(self, text: str) -> Optional[str]:
        if self.use_original_format:
            # Extract the final answer based on ####
            if "####" not in text:
                return None
            parts = text.split("####")
            assert len(parts) >= 2
            return parts[-1].strip()

        text = text.replace(",", "")
        pred_answer = FIND_NUMBERS_REGEX.findall(text)  # TODO: add task to attributes
        if len(pred_answer) == 0:
            return None
        else:
            # Pick the last number
            pred_answer = pred_answer[-1].strip().rstrip(".")
            return pred_answer

    def _grade_answer(self, pred_answer: str, gt_answer: str) -> bool:
        if pred_answer is None:
            return False
        return (
            pred_answer.strip().replace(",", "").lower()
            == gt_answer.replace(",", "").strip().lower()
        )


class GSM8KFormatOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the GSM8K task."""

    def __init__(self, format_reward_positive: bool = True, **_) -> None:
        super().__init__()
        self.is_format_reward_positive = format_reward_positive

    @staticmethod
    def get_format_reward(response: str):
        reasoning_open = '<reasoning>'
        reasoning_close = '</reasoning>'
        answer_open = '<answer>'
        answer_close = '</answer>'
        all_tags = [reasoning_open, reasoning_close, answer_open, answer_close]

        reasoning_open_index = response.find(reasoning_open)
        has_reasoning_open = reasoning_open_index > -1

        reasoning_close_index = response.rfind(reasoning_close)
        has_reasoning_close = reasoning_close_index > -1

        answer_open_index = response.find(answer_open)
        has_answer_open = answer_open_index > -1

        answer_close_index = response.rfind(answer_close)
        has_answer_close = answer_close_index > -1

        is_correct_order_reasoning_tags = False
        is_correct_content_reasoning_tags = False
        reasoning_string = None
        if has_reasoning_open and has_reasoning_close:
            is_correct_order_reasoning_tags = reasoning_open_index < reasoning_close_index
            if is_correct_order_reasoning_tags:
                reasoning_string = response[reasoning_open_index + len(reasoning_open):reasoning_close_index]
                is_correct_content_reasoning_tags = not contains_any(reasoning_string, all_tags)

        is_correct_order_answer_tags = False
        is_correct_content_answer_tags = False
        answer_string = None
        if has_answer_open and has_answer_close:
            is_correct_order_answer_tags = answer_open_index < answer_close_index
            if is_correct_order_answer_tags:
                answer_string = response[answer_open_index + len(answer_open):answer_close_index]
                is_correct_content_answer_tags = not contains_any(answer_string, all_tags)

        is_answer_after_reasoning = is_correct_order_reasoning_tags and is_correct_order_answer_tags \
                                    and reasoning_close_index < answer_open_index
        is_answer_right_after_reasoning = is_answer_after_reasoning \
                                          and response[reasoning_close_index + len(reasoning_close):answer_open_index].strip() == ""

        do_starts_with_reasoning = response.strip().startswith(reasoning_open)
        do_ends_with_answer = response.strip().endswith(answer_close)

        all_checks = [has_reasoning_open, has_reasoning_close, has_answer_open, has_answer_close,
                      is_correct_order_reasoning_tags, is_correct_content_reasoning_tags,
                      is_correct_order_answer_tags, is_correct_content_answer_tags,
                      is_answer_after_reasoning, is_answer_right_after_reasoning, do_starts_with_reasoning,
                      do_ends_with_answer]

        format_reward = sum([int(c) for c in all_checks]) / len(all_checks)
        return format_reward, reasoning_string, answer_string

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        del inputs, batch_size
        predicted_answers = []
        rewards = []
        format_rewards = []
        answer_rewards = []
        reasoning_lengths = []
        answer_lengths = []
        soft_accuracies = []
        hard_accuracies = []

        for resp, ref in zip(responses, references):
            format_reward, reasoning_string, answer_string = self.get_format_reward(resp)
            if not self.is_format_reward_positive:
                format_reward -= 1

            answer_reward = 0
            if answer_string is not None:
                try:
                    answer_reward = int(float(answer_string) == float(ref))
                except ValueError:
                    answer = self._extract_predicted_answer_from_text(answer_string.strip())
                    if answer is not None:
                        answer_reward = 0.5 * int(float(answer) == float(ref))

            predicted_answers.append(answer_string)
            rewards.append(format_reward + answer_reward)
            format_rewards.append(format_reward)
            answer_rewards.append(answer_reward)
            soft_accuracies.append(float(answer_reward > 0))
            hard_accuracies.append(float(answer_reward == 1))
            if reasoning_string is not None:
                reasoning_lengths.append(len(reasoning_string))
            if answer_string is not None:
                answer_lengths.append(len(answer_string))

        return torch.tensor(rewards), {"predicted_answers": predicted_answers,}

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info

    @staticmethod
    def _extract_predicted_answer_from_text(text: str) -> Optional[str]:
        text = text.replace(",", "")
        pred_answer = FIND_NUMBERS_REGEX.findall(text)  # TODO: add task to attributes
        if len(pred_answer) == 0:
            return None
        else:
            # Pick the last number
            pred_answer = pred_answer[-1].strip().rstrip(".")
            return pred_answer

class Multiplication4x4FormatOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for multiplication 4x4 task."""

    def __init__(self, format_reward_positive: bool = True, **_) -> None:
        super().__init__()
        self.is_format_reward_positive = format_reward_positive

    @staticmethod
    def get_format_reward(response: str, gt_step_strings: List[str]):
        reasoning_open = '<reasoning>'
        reasoning_close = '</reasoning>'
        step_open = '<step>'
        step_close = '</step>'
        answer_open = '<answer>'
        answer_close = '</answer>'
        all_tags = [reasoning_open, reasoning_close, answer_open, answer_close]

        reasoning_open_index = response.find(reasoning_open)
        has_reasoning_open = reasoning_open_index > -1

        reasoning_close_index = response.rfind(reasoning_close)
        has_reasoning_close = reasoning_close_index > -1

        answer_open_index = response.find(answer_open)
        has_answer_open = answer_open_index > -1

        answer_close_index = response.rfind(answer_close)
        has_answer_close = answer_close_index > -1

        is_correct_order_reasoning_tags = False
        is_correct_content_reasoning_tags = False
        reasoning_ends_with_step = False
        reasoning_string = None
        step_strings = []
        steps_are_sequential = []
        if has_reasoning_open and has_reasoning_close:
            is_correct_order_reasoning_tags = reasoning_open_index < reasoning_close_index
            if is_correct_order_reasoning_tags:
                reasoning_string = response[reasoning_open_index + len(reasoning_open):reasoning_close_index]
                is_correct_content_reasoning_tags = not contains_any(reasoning_string, all_tags)
                reasoning_substring = reasoning_string.strip()
                reasoning_ends_with_step = reasoning_substring.endswith(step_close)
                while len(reasoning_substring) > 0:
                    step_open_index = reasoning_substring.find(step_open)
                    steps_are_sequential.append(step_open_index == 0)
                    if step_open_index > -1:
                        step_close_index = reasoning_substring.find(step_close)
                        if step_open_index < step_close_index:
                            step_string = reasoning_substring[step_open_index + len(step_open):step_close_index]
                            step_strings.append(step_string.strip())
                            index = step_close_index + len(step_close)
                            reasoning_substring = reasoning_substring[index:].strip()
                            continue

                    break

        are_steps_sequential = 0 if len(steps_are_sequential) == 0 else sum(steps_are_sequential) / len(steps_are_sequential)
        is_correct_number_of_steps = len(step_strings) == len(gt_step_strings)
        correct_step_strings = [False] * len(gt_step_strings)
        for i, (step_string, gt_step_string) in enumerate(zip(step_strings, gt_step_strings)):
            correct_step_strings[i] = step_string == gt_step_string

        is_correct_order_answer_tags = False
        is_correct_content_answer_tags = False
        answer_string = None
        if has_answer_open and has_answer_close:
            is_correct_order_answer_tags = answer_open_index < answer_close_index
            if is_correct_order_answer_tags:
                answer_string = response[answer_open_index + len(answer_open):answer_close_index]
                is_correct_content_answer_tags = not contains_any(answer_string, all_tags)

        is_answer_after_reasoning = is_correct_order_reasoning_tags and is_correct_order_answer_tags \
                                    and reasoning_close_index < answer_open_index
        is_answer_right_after_reasoning = is_answer_after_reasoning \
                                          and response[reasoning_close_index + len(reasoning_close):answer_open_index].strip() == ""

        do_starts_with_reasoning = response.strip().startswith(reasoning_open)
        do_ends_with_answer = response.strip().endswith(answer_close)

        all_checks = [has_reasoning_open, has_reasoning_close, has_answer_open, has_answer_close,
                      is_correct_order_reasoning_tags, is_correct_content_reasoning_tags,
                      is_correct_order_answer_tags, is_correct_content_answer_tags,
                      is_answer_after_reasoning, is_answer_right_after_reasoning, do_starts_with_reasoning,
                      do_ends_with_answer, reasoning_ends_with_step, are_steps_sequential,
                      is_correct_number_of_steps, *correct_step_strings]

        format_reward = 0.25 * sum([float(c) for c in all_checks]) / len(all_checks)
        return format_reward, reasoning_string, answer_string

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        del inputs, batch_size
        predicted_answers = []
        rewards = []
        format_rewards = []
        answer_rewards = []
        reasoning_lengths = []
        answer_lengths = []
        soft_accuracies = []
        hard_accuracies = []

        for resp, ref in zip(responses, references):
            lines = ref.split('\n')
            gt_step_strings = lines[:-1]
            gt_answer = lines[-1]
            format_reward, reasoning_string, answer_string = self.get_format_reward(resp, gt_step_strings)
            if not self.is_format_reward_positive:
                format_reward -= 1

            answer_reward = 0
            if answer_string is not None:
                try:
                    answer_reward = int(float(answer_string) == float(gt_answer))
                except ValueError:
                    answer = self._extract_predicted_answer_from_text(answer_string.strip())
                    if answer is not None:
                        answer_reward = 0.5 * int(float(answer) == float(gt_answer))

            predicted_answers.append(answer_string)
            rewards.append(format_reward + answer_reward)
            format_rewards.append(format_reward)
            answer_rewards.append(answer_reward)
            soft_accuracies.append(float(answer_reward > 0))
            hard_accuracies.append(float(answer_reward == 1))
            if reasoning_string is not None:
                reasoning_lengths.append(len(reasoning_string))
            if answer_string is not None:
                answer_lengths.append(len(answer_string))

        return torch.tensor(rewards), {"predicted_answers": predicted_answers,}

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info

    @staticmethod
    def _extract_predicted_answer_from_text(text: str) -> Optional[str]:
        text = text.replace(",", "")
        pred_answer = FIND_NUMBERS_REGEX.findall(text)  # TODO: add task to attributes
        if len(pred_answer) == 0:
            return None
        else:
            # Pick the last number
            pred_answer = pred_answer[-1].strip().rstrip(".")
            return pred_answer


if __name__ == '__main__':
    oracle = Multiplication4x4FormatOracle()
    inputs = ["""
    Respond in the following format:
    <reasoning>
    <step>...</step>
    ...
    <step>...</step>
    </reasoning>
    <answer>...</answer>
    
    Step by step, multiply the numbers. For each place value of the multiplier, provide the intermediate multiplication result in the form of <step>multiplicand * place value = result</step>. Sum all the intermediate results and provide the final result in the form of <answer>step result + ... + step result = answer</answer>.
    2365 * 4347
    """]

    responses = ["""
    <reasoning> To multiply 2365 by 4347, we will break it down into steps by multiplying 2365 by each digit in the multiplier (4347), then summing the intermediate results.
    <step>2365 * 7 = 16555</step> <step>2365 * 40 = 94600</step> <step>2365 * 300 = 709500</step> <step>2365 * 4000 = 9460000</step>
    Now, we sum the intermediate results: <step>16555 + 94600 + 709500 + 9460000 = 10384055</step> </reasoning> <answer>16555 + 94600 + 709500 + 9460000 = 10384055</answer>
    """]

    references = ["""2365 * 7 = 16555\n2365 * 40 = 94600\n2365 * 300 = 709500\n2365 * 4000 = 9460000\n10280655"""]

    rewards = oracle.get_reward(inputs, responses, references)
    print(rewards)
