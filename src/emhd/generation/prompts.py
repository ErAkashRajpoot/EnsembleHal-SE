"""Prompt templates for each artefact type.

Structured to elicit outputs amenable to cross-model disagreement analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PromptTemplate:
    """A versioned prompt template for a specific artefact type."""
    artefact_type: str
    version: str
    system_message: str
    user_template: str

    def render(self, **kwargs: str) -> str:
        """Render the user template with the given variables."""
        return self.user_template.format(**kwargs)


# ── Prompt Templates ─────────────────────────────────────────────────────────

CODE_TEMPLATE = PromptTemplate(
    artefact_type="code",
    version="v1",
    system_message=(
        "You are an expert software engineer. Generate clean, correct, and "
        "well-documented code. Follow best practices and handle edge cases."
    ),
    user_template=(
        "Complete the following programming task. Provide only the code solution "
        "without any explanations or markdown formatting.\n\n"
        "Task:\n{prompt}"
    ),
)

API_TEMPLATE = PromptTemplate(
    artefact_type="api",
    version="v1",
    system_message=(
        "You are an expert cloud API developer. Generate correct API usage code "
        "that follows official SDK documentation and handles errors properly."
    ),
    user_template=(
        "Write code that uses the specified cloud API to accomplish the following task. "
        "Use only real, documented API methods and parameters. "
        "Provide only the code without explanations.\n\n"
        "Task:\n{prompt}"
    ),
)

TEST_TEMPLATE = PromptTemplate(
    artefact_type="test",
    version="v1",
    system_message=(
        "You are an expert software tester. Generate comprehensive unit tests "
        "that cover normal cases, edge cases, and error conditions."
    ),
    user_template=(
        "Write unit tests for the following code or specification. "
        "Use appropriate assertions and cover edge cases. "
        "Provide only the test code.\n\n"
        "Task:\n{prompt}"
    ),
)

DOC_TEMPLATE = PromptTemplate(
    artefact_type="doc",
    version="v1",
    system_message=(
        "You are an expert technical writer. Generate clear, accurate, and "
        "concise documentation that faithfully describes the code's behavior."
    ),
    user_template=(
        "Write a concise documentation summary for the following code. "
        "Describe what the code does, its parameters, return values, and any "
        "important behavior. Be precise and accurate.\n\n"
        "Code:\n{prompt}"
    ),
)

REVIEW_TEMPLATE = PromptTemplate(
    artefact_type="review",
    version="v1",
    system_message=(
        "You are an expert code reviewer. Provide constructive, accurate feedback "
        "on code quality, correctness, and potential issues."
    ),
    user_template=(
        "Review the following code change (diff). Identify bugs, issues, and "
        "suggest improvements. Be specific about line-level concerns.\n\n"
        "Diff:\n{prompt}"
    ),
)

EXPLANATION_TEMPLATE = PromptTemplate(
    artefact_type="explanation",
    version="v1",
    system_message=(
        "You are an expert software engineer analyzing bug reports. Provide "
        "accurate, technically precise explanations of software defects."
    ),
    user_template=(
        "Based on the following bug report and fix diff, explain what the bug is, "
        "why it occurs, and how the fix addresses it. Be technically precise.\n\n"
        "{prompt}"
    ),
)


_TEMPLATES: Dict[str, PromptTemplate] = {
    "code": CODE_TEMPLATE,
    "api": API_TEMPLATE,
    "test": TEST_TEMPLATE,
    "doc": DOC_TEMPLATE,
    "review": REVIEW_TEMPLATE,
    "explanation": EXPLANATION_TEMPLATE,
}


def get_prompt_template(artefact_type: str) -> PromptTemplate:
    """Get the prompt template for a given artefact type."""
    if artefact_type not in _TEMPLATES:
        raise ValueError(
            f"Unknown artefact_type: {artefact_type}. "
            f"Available: {list(_TEMPLATES.keys())}"
        )
    return _TEMPLATES[artefact_type]
