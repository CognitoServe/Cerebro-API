"""
tests/test_calculator.py
Tests for the AST-based safe math evaluator.
"""
import pytest
from tools.calculator import calculate


class TestCalculate:

    def test_addition(self):
        assert calculate("1 + 2") == 3.0

    def test_subtraction(self):
        assert calculate("10 - 4") == 6.0

    def test_multiplication(self):
        assert calculate("3 * 4") == 12.0

    def test_division(self):
        assert calculate("10 / 4") == 2.5

    def test_floor_division(self):
        assert calculate("10 // 3") == 3.0

    def test_modulo(self):
        assert calculate("10 % 3") == 1.0

    def test_exponentiation(self):
        assert calculate("2 ** 10") == 1024.0

    def test_compound_expression(self):
        assert calculate("2 ** 10 + 100") == 1124.0

    def test_unary_negation(self):
        assert calculate("-5 + 10") == 5.0

    def test_division_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            calculate("1 / 0")

    def test_floor_division_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            calculate("5 // 0")

    def test_exponent_too_large_raises(self):
        with pytest.raises(ValueError, match="Exponent too large"):
            calculate("2 ** 1001")

    def test_empty_expression_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            calculate("")

    def test_invalid_syntax_raises(self):
        with pytest.raises(SyntaxError):
            calculate("2 +* 3")

    def test_returns_float(self):
        result = calculate("42")
        assert isinstance(result, float)

    def test_nested_expression(self):
        assert calculate("(2 + 3) * (4 - 1)") == 15.0
