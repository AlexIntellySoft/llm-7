"""Проверяет пары учителя и собирает data/{ecql_dataset,train,test}.jsonl (80/20).

Каждый эталон обязан разбираться грамматикой ECQL, использовать только поля своей
сущности и не содержать SQL. Разбиение стратифицировано по уровню сложности и сущности,
чтобы в тесте были и простые, и вложенные запросы по всем четырём сущностям.

    python -m tools.build_dataset
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ecql import SQL_PATTERN, conditions, parse, schema_errors  # noqa: E402
from tools.teacher_pairs import PAIRS  # noqa: E402

INSTRUCTION = "Переведи запрос на ECQL"
TEST_SHARE = 0.2
SEED = 42


def validate() -> list[dict]:
    rows, problems, seen = [], [], set()
    for level, ru, ecql in PAIRS:
        try:
            q = parse(ecql)
        except ValueError as e:
            problems.append(f"{ecql}: {e}")
            continue
        problems += [f"{ecql}: {e}" for e in schema_errors(q)]
        if SQL_PATTERN.search(ecql):
            problems.append(f"{ecql}: похоже на SQL")
        if ru in seen:
            problems.append(f"дубль: {ru}")
        seen.add(ru)
        rows.append({"instruction": INSTRUCTION, "input": ru, "output": ecql,
                     "level": level, "entity": q.entity})
    if problems:
        raise SystemExit("Ошибки в датасете:\n" + "\n".join(problems))
    return rows


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["level"], r["entity"])].append(r)
    rng = random.Random(SEED)
    train, test, carry = [], [], 0.0
    for key in sorted(groups):
        items = groups[key][:]
        rng.shuffle(items)
        # дробные доли переносятся между группами, чтобы в сумме вышло ровно 20%
        carry += len(items) * TEST_SHARE
        k = int(carry)
        carry -= k
        test += items[:k]
        train += items[k:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    rows = validate()
    train, test = split(rows)
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    write(data / "ecql_dataset.jsonl", rows)
    write(data / "train.jsonl", train)
    write(data / "test.jsonl", test)

    ops = Counter(c.op for r in rows for c in conditions(parse(r["output"]).where))
    fmts = Counter(parse(r["output"]).fmt or "—" for r in rows)
    print(f"всего {len(rows)}: train {len(train)}, test {len(test)}")
    for name, part in (("всего", rows), ("train", train), ("test", test)):
        print(f"{name:6}", dict(sorted(Counter(r["level"] for r in part).items())),
              dict(sorted(Counter(r["entity"] for r in part).items())))
    print("операторы:", dict(ops))
    print("форматы AS:", dict(fmts))


if __name__ == "__main__":
    main()
