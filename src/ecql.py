"""Спецификация ECQL: схема сущностей, парсер, нормализация и проверка запросов.

Грамматика (правила задания + две договорённости, без которых датасет не согласовать):

    query     := FETCH [ENTITY] [WHERE expr] [AS FORMAT]
    expr      := and_expr ( || and_expr )*
    and_expr  := primary ( && primary )*
    primary   := ( expr ) | condition
    condition := @field OP value
    OP        := IS | NOT | ABOVE | BELOW
    value     := 'строка' | число
    FORMAT    := JSON | TABLE | LIST

Договорённости: WHERE стоит перед условиями (так в примере из задания); вложенность —
это скобки. && связывает сильнее ||, но в датасете при смешении связок скобки ставятся
всегда, чтобы модели не приходилось помнить приоритеты.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ENTITIES: dict[str, dict[str, str]] = {
    # поле -> тип значения ("str" или "num")
    "EMPLOYEES": {
        "name": "str", "city": "str", "department": "str", "position": "str",
        "salary": "num", "age": "num", "experience": "num", "status": "str",
    },
    "PROJECTS": {
        "name": "str", "status": "str", "budget": "num", "priority": "str",
        "manager": "str", "team_size": "num", "client": "str", "year": "num",
    },
    "INVENTORY": {
        "item": "str", "category": "str", "quantity": "num", "price": "num",
        "warehouse": "str", "status": "str",
    },
    "DEALS": {
        "client": "str", "amount": "num", "stage": "str", "region": "str",
        "manager": "str", "probability": "num", "year": "num",
    },
}
OPERATORS = ("IS", "NOT", "ABOVE", "BELOW")
FORMATS = ("JSON", "TABLE", "LIST")

# Признаки стандартного SQL в ответе модели — критерий «галлюцинации» из задания
SQL_PATTERN = re.compile(
    r"\b(SELECT|FROM|AND|OR|LIKE|ORDER\s+BY|GROUP\s+BY|LIMIT)\b|!=|<>|>=|<=|(?<![!<>])=|[<>]",
)

_TOKEN = re.compile(
    r"\s*(?:(?P<entity>\[[A-Z_]+\])|(?P<field>@[a-z_]+)|(?P<str>'[^']*')"
    r"|(?P<num>-?\d+(?:\.\d+)?)|(?P<op>&&|\|\||\(|\))|(?P<word>[A-Za-z_]+)|(?P<bad>\S))"
)


class ECQLSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    field: str
    op: str
    value: str | float


@dataclass(frozen=True)
class BoolExpr:
    op: str  # "&&" или "||"
    items: tuple


@dataclass(frozen=True)
class Query:
    entity: str
    where: object | None  # Condition | BoolExpr | None
    fmt: str | None


def tokenize(text: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            break
        kind = m.lastgroup
        if kind == "bad":
            raise ECQLSyntaxError(f"недопустимый символ {m.group(kind)!r}")
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, text: str):
        self.tokens = tokenize(text)
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def take(self, kind=None, value=None):
        tok = self.peek()
        if tok[0] is None or (kind and tok[0] != kind) or (value and tok[1] != value):
            want = value or kind
            raise ECQLSyntaxError(f"ожидалось {want}, получено {tok[1]!r}")
        self.i += 1
        return tok[1]

    def query(self) -> Query:
        self.take("word", "FETCH")
        entity = self.take("entity")[1:-1]
        if entity not in ENTITIES:
            raise ECQLSyntaxError(f"неизвестная сущность [{entity}]")
        where = fmt = None
        if self.peek() == ("word", "WHERE"):
            self.take()
            where = self.expr()
        if self.peek() == ("word", "AS"):
            self.take()
            fmt = self.take("word")
            if fmt not in FORMATS:
                raise ECQLSyntaxError(f"неизвестный формат {fmt}")
        if self.peek()[0] is not None:
            raise ECQLSyntaxError(f"лишний текст после запроса: {self.peek()[1]!r}")
        return Query(entity, where, fmt)

    def _chain(self, sep, sub):
        items = [sub()]
        while self.peek() == ("op", sep):
            self.take()
            items.append(sub())
        return items[0] if len(items) == 1 else BoolExpr(sep, tuple(items))

    def expr(self):
        return self._chain("||", self.and_expr)

    def and_expr(self):
        return self._chain("&&", self.primary)

    def primary(self):
        if self.peek() == ("op", "("):
            self.take()
            inner = self.expr()
            self.take("op", ")")
            return inner
        field = self.take("field")[1:]
        op = self.take("word")
        if op not in OPERATORS:
            raise ECQLSyntaxError(f"неизвестный оператор {op}")
        kind, raw = self.peek()
        if kind == "str":
            value = raw[1:-1]
        elif kind == "num":
            value = float(raw)
        else:
            raise ECQLSyntaxError(f"ожидалось значение, получено {raw!r}")
        self.take()
        return Condition(field, op, value)


def parse(text: str) -> Query:
    """Разбирает запрос; бросает ECQLSyntaxError, если он не по грамматике."""
    return _Parser(text).query()


def conditions(node) -> list[Condition]:
    if node is None:
        return []
    if isinstance(node, Condition):
        return [node]
    return [c for item in node.items for c in conditions(item)]


def schema_errors(q: Query) -> list[str]:
    """Поля, которых нет у сущности, и несовпадение типа значения с типом поля."""
    fields, errors = ENTITIES[q.entity], []
    for c in conditions(q.where):
        if c.field not in fields:
            errors.append(f"у [{q.entity}] нет поля @{c.field}")
            continue
        is_num = isinstance(c.value, float)
        if fields[c.field] == "num" and not is_num:
            errors.append(f"@{c.field} — число, а значение строка")
        if fields[c.field] == "str" and is_num:
            errors.append(f"@{c.field} — строка, а значение число")
        if c.op in ("ABOVE", "BELOW") and not is_num:
            errors.append(f"{c.op} применён к строке @{c.field}")
    return errors


def canonical(node):
    """Форма выражения, не зависящая от порядка операндов внутри && и ||."""
    if node is None or isinstance(node, Condition):
        return node
    flat = []
    for item in node.items:
        item = canonical(item)
        if isinstance(item, tuple) and item[0] == node.op:
            flat.extend(item[1])  # (a && b) && c == a && b && c
        else:
            flat.append(item)
    return (node.op, frozenset(flat))


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())
