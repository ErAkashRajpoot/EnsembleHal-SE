"""Offline synthetic LLM client — no API calls required.

Generates realistic, varied code outputs locally using template mutations
and controlled randomisation. Each "model" uses a different code style seed
to produce genuinely different outputs, enabling meaningful cross-model
disagreement analysis for the EMHD pipeline.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import random
import re
import textwrap
from typing import Any, Dict, List, Optional

from . import BaseLLMClient, GenerationResult

logger = logging.getLogger(__name__)


# ── Style profiles — each simulated model writes code differently ──────────
_MODEL_STYLES: Dict[str, Dict[str, Any]] = {
    "offline/model-alpha": {
        "docstring_style": "google",
        "naming": "snake_case",
        "error_handling": "try_except",
        "verbosity": 0.8,
        "hallucination_rate": 0.15,
        "imports_style": "explicit",
    },
    "offline/model-beta": {
        "docstring_style": "numpy",
        "naming": "camelCase",
        "error_handling": "assert",
        "verbosity": 0.5,
        "hallucination_rate": 0.25,
        "imports_style": "star",
    },
    "offline/model-gamma": {
        "docstring_style": "sphinx",
        "naming": "snake_case",
        "error_handling": "if_check",
        "verbosity": 0.9,
        "hallucination_rate": 0.10,
        "imports_style": "explicit",
    },
    "offline/model-delta": {
        "docstring_style": "plain",
        "naming": "snake_case",
        "error_handling": "try_except",
        "verbosity": 0.3,
        "hallucination_rate": 0.30,
        "imports_style": "minimal",
    },
    "offline/model-epsilon": {
        "docstring_style": "google",
        "naming": "camelCase",
        "error_handling": "raise",
        "verbosity": 0.6,
        "hallucination_rate": 0.20,
        "imports_style": "explicit",
    },
}

# ── Building blocks for synthetic code generation ──────────────────────────

_STDLIB_MODULES = [
    "os", "sys", "math", "json", "re", "collections", "itertools",
    "functools", "typing", "pathlib", "datetime", "hashlib", "random",
    "string", "io", "abc", "dataclasses", "logging", "unittest",
]

_FAKE_APIS = [
    "cloud_storage.upload_blob", "api_gateway.invoke_endpoint",
    "auth_service.verify_token", "message_queue.publish_event",
    "cache_layer.invalidate", "ml_pipeline.run_inference",
    "database.execute_query", "notification.send_alert",
]

_HALLUCINATION_PATTERNS = [
    # Invented method names
    lambda fn: fn + ".optimized_v2()",
    lambda fn: fn.replace("def ", "def _internal_") + "  # deprecated API",
    # Wrong parameter names
    lambda fn: fn.replace("(", "(enable_turbo=True, "),
    # Fabricated library calls
    lambda fn: f"import torch.quantum\ntorch.quantum.entangle({fn})",
    # Non-existent built-in
    lambda fn: f"result = supereval({fn})",
]


def _deterministic_seed(model_id: str, prompt: str, sample_idx: int) -> int:
    """Create a reproducible seed from model + prompt + sample index."""
    h = hashlib.md5(f"{model_id}::{prompt[:200]}::{sample_idx}".encode()).hexdigest()
    return int(h[:8], 16)


def _extract_function_name(prompt: str) -> str:
    """Try to extract a function name from the prompt."""
    # Look for "def function_name" patterns
    match = re.search(r'def\s+(\w+)', prompt)
    if match:
        return match.group(1)
    # Look for quoted function names
    match = re.search(r'["\'](\w+)["\']', prompt)
    if match:
        return match.group(1)
    # Fallback
    words = re.findall(r'\w+', prompt[:100])
    return "_".join(words[:3]).lower() if words else "solution"


def _generate_code_body(
    prompt: str,
    style: Dict[str, Any],
    rng: random.Random,
    temperature: float,
) -> str:
    """Generate a synthetic code solution with model-specific style."""
    func_name = _extract_function_name(prompt)
    
    # Pick imports based on style
    n_imports = rng.randint(1, 4)
    selected_imports = rng.sample(_STDLIB_MODULES, min(n_imports, len(_STDLIB_MODULES)))
    import_lines = [f"import {m}" for m in selected_imports]
    
    # Rename to camelCase if that's the style
    if style["naming"] == "camelCase":
        parts = func_name.split("_")
        func_name = parts[0] + "".join(p.capitalize() for p in parts[1:])
    
    # Generate parameter list based on prompt analysis
    param_words = re.findall(r'\b(?:array|list|string|number|integer|value|data|input|nums?|arr|n|k|target)\b', prompt.lower())
    params = list(dict.fromkeys(param_words))[:4] or ["data"]
    param_str = ", ".join(params)
    
    # Docstring
    docstrings = {
        "google": f'    """Solve the task as specified.\n\n    Args:\n        {params[0]}: Input parameter.\n\n    Returns:\n        Computed result.\n    """',
        "numpy": f'    """\n    Parameters\n    ----------\n    {params[0]} : any\n        Input parameter.\n\n    Returns\n    -------\n    result\n        The computed output.\n    """',
        "sphinx": f'    """Solve the task.\n\n    :param {params[0]}: Input parameter.\n    :returns: Computed result.\n    """',
        "plain": f'    """Solution for: {func_name}."""',
    }
    docstring = docstrings.get(style["docstring_style"], docstrings["plain"])
    
    # Body logic — vary based on temperature and style
    body_templates = [
        # Template 1: Iterative
        f"""    result = []
    for i in range(len({params[0]})):
        item = {params[0]}[i]
        if isinstance(item, (int, float)):
            result.append(item * 2)
        else:
            result.append(item)
    return result""",
        # Template 2: Recursive
        f"""    if not {params[0]}:
        return {params[0]}
    head = {params[0]}[0]
    tail = {func_name}({params[0]}[1:])
    return [head] + tail""",
        # Template 3: Dict-based
        f"""    seen = {{}}
    output = []
    for val in {params[0]}:
        key = str(val)
        if key not in seen:
            seen[key] = True
            output.append(val)
    return output""",
        # Template 4: Comprehension
        f"""    return [x for x in {params[0]} if x is not None]""",
        # Template 5: Math-heavy
        f"""    total = sum({params[0]}) if hasattr({params[0]}, '__iter__') else {params[0]}
    count = len({params[0]}) if hasattr({params[0]}, '__len__') else 1
    if count == 0:
        return 0
    mean = total / count
    variance = sum((x - mean) ** 2 for x in {params[0]}) / count
    return round(variance ** 0.5, 4)""",
        # Template 6: String processing
        f"""    if isinstance({params[0]}, str):
        tokens = {params[0]}.split()
        return ' '.join(sorted(set(tokens)))
    return str({params[0]})""",
    ]
    
    # Select template based on seed + temperature jitter
    template_idx = rng.randint(0, len(body_templates) - 1)
    body = body_templates[template_idx]
    
    # Error handling wrapper
    if style["error_handling"] == "try_except":
        body = f"    try:\n    {body}\n    except Exception as e:\n        raise ValueError(f'Error in {func_name}: {{e}}')"
    elif style["error_handling"] == "assert":
        body = f"    assert {params[0]} is not None, '{params[0]} must not be None'\n{body}"
    elif style["error_handling"] == "if_check":
        body = f"    if {params[0]} is None:\n        return None\n{body}"
    
    # Inject hallucinations based on style's hallucination_rate
    if rng.random() < style["hallucination_rate"] * (1 + temperature):
        hallucination = rng.choice(_HALLUCINATION_PATTERNS)
        hallucinated_line = hallucination(func_name)
        body += f"\n    # Additional optimization\n    {hallucinated_line}"
    
    # Assemble
    lines = import_lines + ["", f"def {func_name}({param_str}):", docstring, body]
    
    # Add extra verbosity (comments) based on style
    if rng.random() < style["verbosity"]:
        lines.insert(len(import_lines), f"# Solution generated for: {func_name}")
        lines.insert(len(import_lines) + 1, f"# Approach: {'iterative' if template_idx < 3 else 'functional'}")
    
    return "\n".join(lines)


def _generate_review(prompt: str, style: Dict[str, Any], rng: random.Random) -> str:
    """Generate a synthetic code review."""
    issues = [
        "Missing error handling for edge case with empty input",
        "Variable name 'x' is not descriptive enough",
        "Consider using a dictionary for O(1) lookups instead of list iteration",
        "The function lacks type hints which hurts readability",
        "Potential off-by-one error in the loop boundary",
        "Memory usage could be high for large inputs — consider generators",
        "Missing docstring describing return value",
        "This could be simplified with a list comprehension",
        "Thread safety concern: shared mutable state without locks",
        "API key should not be hardcoded — use environment variables",
    ]
    n_issues = rng.randint(2, 5)
    selected = rng.sample(issues, min(n_issues, len(issues)))
    
    severity = ["🔴 Critical", "🟡 Warning", "🟢 Suggestion"]
    review_lines = ["## Code Review\n"]
    for i, issue in enumerate(selected):
        sev = rng.choice(severity)
        review_lines.append(f"{i+1}. **{sev}**: {issue}")
    
    if rng.random() < style["hallucination_rate"]:
        review_lines.append(f"{len(selected)+1}. **🔴 Critical**: Uses deprecated `asyncio.coroutine` decorator (removed in Python 3.11)")
    
    return "\n".join(review_lines)


def _generate_doc(prompt: str, style: Dict[str, Any], rng: random.Random) -> str:
    """Generate synthetic documentation."""
    func_name = _extract_function_name(prompt)
    
    doc = f"""# `{func_name}`

## Description
This function processes the given input and returns the computed result.
It handles various edge cases including empty inputs and type mismatches.

## Parameters
- `data`: The primary input to process. Accepts lists, strings, or numeric types.

## Returns
- The processed result matching the expected output format.

## Example
```python
result = {func_name}([1, 2, 3])
print(result)  # Expected output
```

## Notes
- Time complexity: O(n) where n is the input size.
- Space complexity: O(n) for the result buffer.
"""
    if rng.random() < style["hallucination_rate"]:
        doc += f"\n## Deprecated\nThis function replaces the legacy `{func_name}_v1` which was removed in v2.3.\n"
    
    return doc


def _generate_explanation(prompt: str, style: Dict[str, Any], rng: random.Random) -> str:
    """Generate a synthetic bug explanation."""
    explanations = [
        "The bug occurs due to an off-by-one error in the loop boundary condition.",
        "The root cause is a missing null check before dereferencing the pointer.",
        "The issue stems from integer overflow when computing large values.",
        "The bug is caused by incorrect operator precedence in the conditional expression.",
        "The fix addresses a race condition in the concurrent access pattern.",
    ]
    fix_approaches = [
        "The fix adds a boundary check before the array access.",
        "The patch corrects the comparison operator from '<' to '<='.",
        "The fix introduces proper synchronization using a mutex lock.",
        "The resolution adds input validation at the function entry point.",
    ]
    
    explanation = rng.choice(explanations)
    fix = rng.choice(fix_approaches)
    
    result = f"## Bug Analysis\n\n{explanation}\n\n## Fix Explanation\n\n{fix}\n"
    
    if rng.random() < style["hallucination_rate"]:
        result += "\n## Additional Context\nThis is a known issue tracked in CVE-2024-XXXXX (fabricated reference).\n"
    
    return result


class OfflineClient(BaseLLMClient):
    """Synthetic LLM client that generates outputs locally without API calls.
    
    Uses template-based generation with model-specific style profiles to
    produce varied outputs suitable for cross-model disagreement analysis.
    Zero cost, zero latency, no rate limits.
    """

    def __init__(self, model_id: str, provider: str = "offline", api_key: str = "", **kwargs: Any):
        super().__init__(model_id=model_id, provider=provider, api_key=api_key, **kwargs)
        self.style = _MODEL_STYLES.get(model_id, _MODEL_STYLES["offline/model-alpha"])
        self._call_count = 0

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        logprobs: bool = False,
        n: int = 1,
        **kwargs: Any,
    ) -> List[GenerationResult]:
        self._call_count += 1
        results = []
        
        for i in range(n):
            seed = _deterministic_seed(self.model_id, prompt, self._call_count * 100 + i)
            rng = random.Random(seed)
            
            # Detect artefact type from prompt keywords
            prompt_lower = prompt.lower()
            if any(kw in prompt_lower for kw in ["review", "diff", "code change"]):
                text = _generate_review(prompt, self.style, rng)
            elif any(kw in prompt_lower for kw in ["documentation", "docstring", "describe"]):
                text = _generate_doc(prompt, self.style, rng)
            elif any(kw in prompt_lower for kw in ["bug report", "fix diff", "defect"]):
                text = _generate_explanation(prompt, self.style, rng)
            else:
                text = _generate_code_body(prompt, self.style, rng, temperature)
            
            # Truncate to approximate max_tokens (rough estimate: 4 chars/token)
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                text = text[:max_chars]
            
            # Simulate token counts
            input_tokens = len(prompt.split()) + 50  # prompt + system message
            output_tokens = len(text.split())
            
            # Generate fake logprobs if requested
            fake_logprobs = None
            if logprobs:
                tokens = text.split()[:20]
                fake_logprobs = {
                    "tokens": tokens,
                    "token_logprobs": [rng.uniform(-3.0, -0.01) for _ in tokens],
                }
            
            results.append(GenerationResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                logprobs=fake_logprobs,
                finish_reason="stop",
                model_id=self.model_id,
                provider=self.provider,
                temperature=temperature,
            ))
        
        return results

    async def close(self) -> None:
        pass  # Nothing to clean up
