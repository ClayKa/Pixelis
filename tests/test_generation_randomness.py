"""Checks for diversity in already generated detail-perception data.

These tests validate artifacts produced by the data-generation pipeline. They
are skipped when the generated JSONL file is not present, so a clean checkout can
still run the unit test suite.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from core.data_generation.detail_perception import DetailPerceptionTaskGenerator


GENERATED_FILE = Path("data_outputs/specialized/detail_perception_task.jsonl")


@pytest.fixture(scope="module")
def generated_questions() -> list[str]:
    if not GENERATED_FILE.exists():
        pytest.skip(f"Generated data artifact not found: {GENERATED_FILE}")

    questions: list[str] = []
    with GENERATED_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            question = data.get("question")
            assert question, f"Missing question field at line {line_number}"
            questions.append(question)

    assert questions, f"No questions found in {GENERATED_FILE}"
    return questions


@pytest.mark.requires_generated_data
def test_generated_questions_do_not_have_exact_duplicates(generated_questions: list[str]) -> None:
    duplicates = [
        question
        for question, count in Counter(generated_questions).items()
        if count > 1
    ]

    assert not duplicates, f"Found duplicate generated questions: {duplicates[:3]}"


@pytest.mark.requires_generated_data
def test_generated_questions_have_diverse_beginnings(generated_questions: list[str]) -> None:
    beginnings = Counter(question[:30] for question in generated_questions)
    diversity_ratio = len(beginnings) / len(generated_questions)

    assert diversity_ratio > 0.5
    assert beginnings.most_common(1)[0][1] <= max(3, len(generated_questions) * 0.25)


@pytest.mark.requires_generated_data
def test_generated_questions_use_framing_templates(generated_questions: list[str]) -> None:
    templates = DetailPerceptionTaskGenerator.QUESTION_FRAMING_TEMPLATES

    template_matches = 0
    for question in generated_questions:
        for template in templates:
            template_start = template.split("{", maxsplit=1)[0].strip()
            if template_start and question.startswith(template_start[:20]):
                template_matches += 1
                break

    assert template_matches >= len(generated_questions) * 0.5
