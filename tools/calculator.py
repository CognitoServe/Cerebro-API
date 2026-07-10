"""
tools/calculator.py
Safe math evaluator using AST walking — no exec(), no eval() of raw strings.
Only allows numeric literals, binary operations (+−×÷^), and unary negation.
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Union

# ---------------------------------------------------------------------------
# Allowed operations
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[type, Any] = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.Pow:      operator.pow,
    ast.Mod:      operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPS: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate a safe AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left  = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("Division by zero in expression")
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large (> 1000) — refusing to compute")
        return op_fn(left, right)

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))

    raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def calculate(expression: str) -> float:
    """
    Safely evaluate a numeric expression string.

    Supports: +  −  *  /  //  %  **  unary −
    Does NOT support: function calls, variables, string ops, imports, exec, eval.

    Args:
        expression: A math expression string, e.g. "2**10 + 100"

    Returns:
        The numeric result as a float.

    Raises:
        ValueError:        On unsupported syntax or operators.
        ZeroDivisionError: On division by zero.
        SyntaxError:       If the expression cannot be parsed.
    """
    if not expression or not expression.strip():
        raise ValueError("expression must be a non-empty string")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise SyntaxError(f"Invalid expression syntax: {exc}") from exc

    result = _eval_node(tree.body)
    return float(result)
