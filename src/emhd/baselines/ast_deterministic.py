"""AST Deterministic Analysis baseline.

Uses static analysis (AST parsing) to detect hallucinations without ML.
Checks for undefined variables, invalid API calls, syntax errors, etc.
"""
from __future__ import annotations

import ast
import re
from typing import Dict, List, Set


class ASTDeterministicBaseline:
    """Deterministic AST-based hallucination detection."""

    def _check_python_syntax(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _extract_defined_names(self, code: str) -> Set[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
        return names

    def _extract_used_names(self, code: str) -> Set[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        return names

    def predict_single(self, code: str) -> float:
        """Score: higher = more likely hallucinated. Combines multiple signals."""
        signals = []
        # Syntax check
        if not self._check_python_syntax(code):
            return 1.0
        # Undefined variable ratio
        defined = self._extract_defined_names(code) | {"print","len","range","int","str","float","list","dict","set","tuple","type","isinstance","open","super","self","cls","True","False","None","Exception","ValueError","TypeError","KeyError","IndexError"}
        used = self._extract_used_names(code)
        undefined = used - defined
        if used:
            signals.append(len(undefined) / len(used))
        else:
            signals.append(0.0)
        # Empty or trivial code
        lines = [l.strip() for l in code.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
        if len(lines) < 2:
            signals.append(0.5)
        else:
            signals.append(0.0)
        return float(min(max(sum(signals) / len(signals), 0.0), 1.0))

    def predict_batch(self, codes: List[str]) -> List[float]:
        return [self.predict_single(c) for c in codes]
