"""Метрики на тестовой выборке: три критерия задания плюс точное совпадение.

1. Синтаксис — ответ разбирается грамматикой ECQL (@, [], &&/||, скобки, кавычки).
2. Логика — та же сущность, тот же формат AS, те же условия (поле, оператор, значение)
   и та же структура И/ИЛИ; порядок операндов внутри && и || не важен.
3. Галлюцинации — в ответе есть элементы стандартного SQL (SELECT, FROM, AND, =, >…).
"""
from __future__ import annotations

from collections import Counter

from .ecql import (
    SQL_PATTERN, ECQLSyntaxError, canonical, conditions, normalize_spaces, parse, schema_errors,
)


def score_one(reference: str, prediction: str) -> dict:
    ref = parse(reference)
    row = {
        "exact_match": normalize_spaces(reference) == normalize_spaces(prediction),
        "sql_hallucination": bool(SQL_PATTERN.search(prediction)),
        "syntax_valid": False, "schema_valid": False, "entity_correct": False,
        "format_correct": False, "conditions_correct": False, "logic_correct": False,
        "operator_errors": 0, "error": "",
    }
    try:
        pred = parse(prediction)
    except ECQLSyntaxError as e:
        row["error"] = f"синтаксис: {e}"
        return row
    row["syntax_valid"] = True
    errors = schema_errors(pred)
    row["schema_valid"] = not errors
    row["entity_correct"] = pred.entity == ref.entity
    row["format_correct"] = pred.fmt == ref.fmt
    ref_conds, pred_conds = Counter(conditions(ref.where)), Counter(conditions(pred.where))
    row["conditions_correct"] = ref_conds == pred_conds
    row["logic_correct"] = (
        row["entity_correct"] and row["format_correct"]
        and canonical(ref.where) == canonical(pred.where)
    )
    # Перепутанный оператор: поле и значение совпали с эталоном, а оператор — нет
    ref_keys = {(c.field, c.value): c.op for c in ref_conds}
    row["operator_errors"] = sum(
        1 for c in pred_conds
        if (c.field, c.value) in ref_keys and ref_keys[(c.field, c.value)] != c.op
    )
    problems = list(errors)
    if not row["entity_correct"]:
        problems.append(f"сущность [{pred.entity}] вместо [{ref.entity}]")
    if not row["format_correct"]:
        problems.append(f"формат {pred.fmt} вместо {ref.fmt}")
    if not row["conditions_correct"]:
        problems.append("условия отличаются")
    elif not row["logic_correct"] and row["entity_correct"] and row["format_correct"]:
        problems.append("условия те же, но иначе сгруппированы")
    row["error"] = "; ".join(problems)
    return row


METRIC_NAMES = {
    "syntax_valid": "синтаксис корректен",
    "schema_valid": "поля и типы есть в схеме",
    "entity_correct": "верная сущность",
    "conditions_correct": "верные условия",
    "format_correct": "верный формат AS",
    "logic_correct": "логически эквивалентен эталону",
    "exact_match": "точное совпадение строки",
    "sql_hallucination": "есть элементы SQL",
}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {name: sum(r[name] for r in rows) / n for name in METRIC_NAMES}
    out["operator_errors_total"] = sum(r["operator_errors"] for r in rows)
    out["n"] = n
    return out
