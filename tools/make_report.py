"""Собирает notebooks/report.ipynb. Ноутбук читает только файлы из data/ и outputs/
(результаты Colab-прогона), поэтому выполняется локально без GPU:

    .venv/bin/python -m tools.make_report
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

GITHUB = "https://github.com/AlexIntellySoft/llm-7"
COLAB = "https://colab.research.google.com/github/AlexIntellySoft/llm-7/blob/main/notebooks/train_colab.ipynb"
HF = "https://huggingface.co/Alex-Popov/qwen3-4b-ecql-lora"

cells = [
md(f"""# DSL Query Generator: дообучение Qwen3-4B переводу с русского на ECQL

Малая модель **Qwen3-4B-Instruct-2507** (4-bit) дообучена LoRA-адаптером переводить запросы
на русском языке во внутренний язык запросов ECQL вымышленной компании «E-Corp».

* Код и данные: [{GITHUB}]({GITHUB})
* Ноутбук обучения в Colab: [`notebooks/train_colab.ipynb`]({COLAB})
* Веса адаптеров: [r=16]({HF}-r16) (основной) и [r=8]({HF}-r8)
* Метрики и ответы модели на тестовой выборке: `outputs/metrics.json`,
  `outputs/predictions_r16.jsonl`, `outputs/predictions_r8.jsonl`

Этот ноутбук ничего не обучает. Он читает результаты Colab-прогона из `outputs/` и
выполняется без GPU."""),
code("""import json, sys
from collections import Counter
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src.ecql import ENTITIES, conditions, parse
from src.evaluate import METRIC_NAMES

def jsonl(p): return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if l]
pd.set_option("display.max_colwidth", None)
data, train, test = jsonl("data/ecql_dataset.jsonl"), jsonl("data/train.jsonl"), jsonl("data/test.jsonl")
metrics = json.loads((ROOT / "outputs/metrics.json").read_text())
preds = {n: jsonl(f"outputs/predictions_{n}.jsonl") for n in ("r8", "r16")}
hist = {n: json.loads((ROOT / f"outputs/log_history_{n}.json").read_text()) for n in ("r8", "r16")}"""),

md("""## 1. Спецификация ECQL и датасет

Правила задания: `FETCH`, сущность в `[]`, поля с `@`, операторы `IS / NOT / ABOVE / BELOW`,
связки `&& / ||`, формат `AS JSON / TABLE / LIST`. Без двух договорённостей датасет не сделать
согласованным, поэтому я их добавил:

* условия начинаются с `WHERE` (так в примере из задания);
* «вложенность» означает скобки: при смешении `&&` и `||` скобки ставятся всегда.

Ещё для каждой сущности задана **схема**: набор полей и их типы. Строковые значения пишутся
по-английски («Москва» → `'Moscow'`, «заморожен» → `'on_hold'`), числа полностью
(«2,5 млн» → `2500000`). Полная схема и промпт учителю: `docs/teacher_prompt.md`.

**Модель-учитель — Claude.** Ключ YandexGPT из прошлых заданий истёк. Каждый эталон
автоматически проверен парсером `src/ecql.py`: грамматика, поля сущности, типы значений,
отсутствие SQL. Разбиение 80/20 стратифицировано по сложности и сущности."""),
code("""df = pd.DataFrame(data)
order = ["all", "simple", "and", "or", "nested"]
tab = pd.crosstab(df["entity"], df["level"])[order]
tab.loc["итого"] = tab.sum()
tab["итого"] = tab.sum(axis=1)
print(f"всего {len(data)} пар: train {len(train)}, test {len(test)}")
print("уровни: all — без условий, simple — одно условие, and — 2–3 через &&, or — через ||, nested — && и || со скобками")
tab"""),
code("""ops = Counter(c.op for r in data for c in conditions(parse(r["output"]).where))
fmts = Counter(parse(r["output"]).fmt or "без AS" for r in data)
print("операторы:", dict(ops))
print("формат вывода:", dict(fmts))
print("поля:", {e: len(f) for e, f in ENTITIES.items()})
df.groupby("level").head(1)[["level", "input", "output"]]"""),

md("""## 2. Базовая модель и LoRA

| параметр | значение | почему |
| --- | --- | --- |
| модель | `Qwen/Qwen3-4B-Instruct-2507` | из рекомендованных в задании; в 4-bit занимает ≈3 ГБ на T4; instruct-версия без «размышлений» `<think>`, хорошо понимает русский |
| квантование | 4-bit NF4 + double quant, вычисления в fp16 | QLoRA-схема из задания; T4 не поддерживает bf16 |
| `r` | **16** (и контрольный прогон 8) | задача узкая (≈30 полей, 4 оператора), а данных мало (148 примеров). r 8–16 хватает, чтобы выучить формат; больший ранг на таком объёме скорее запомнит примеры, чем обобщит |
| `lora_alpha` | **2·r** (32 и 16) | стандартный масштаб alpha/r = 2. Он одинаков в обоих прогонах, поэтому они различаются только рангом |
| целевые модули | `q_proj, k_proj, v_proj, o_proj` | рекомендация задания. Перевод здесь — это в основном «какое слово запроса соответствует какому полю и оператору», а такие связи задаёт внимание. MLP не трогаем, чтобы адаптер остался маленьким |
| `lora_dropout` | 0.05 | лёгкая регуляризация при 148 примерах |
| обучаемых параметров | 5.9 млн (r=16), 2.9 млн (r=8) | 0.15% и 0.07% от 4 млрд |

Loss считается **только по ответу** (формат prompt/completion в `SFTTrainer`). Модель учится
писать ECQL, а не повторять русский запрос."""),

md("""## 3. Дообучение

`trl.SFTTrainer`: lr 2e-4 (косинусное расписание, 3 шага прогрева), 3 эпохи, батч 2 ×
накопление градиента 4 = 8, `max_length` 256 (самый длинный пример — 113 токенов), fp16.
Всего 57 шагов, на T4 около 3 минут на прогон.

Loss на тестовой выборке считался каждые 4 шага, только для наблюдения: в обучении она не
участвует и чекпоинт по ней не выбирался.

*Практическая деталь.* Для 4-bit модели `SFTTrainer` переводит веса LoRA в bfloat16
(рекомендация QLoRA для A100/H100). На T4 это роняет fp16-обучение ошибкой
`_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'`. Поэтому
после создания тренера обучаемые веса возвращаются в fp32 (`src/finetune.py`)."""),
code("""display(Image(str(ROOT / "outputs/loss_curve.png")))
rows = []
for n, h in hist.items():
    tr = {x["step"]: x["loss"] for x in h if "loss" in x}
    ev = {x["step"]: x["eval_loss"] for x in h if "eval_loss" in x}
    run = next(x for x in h if "train_runtime" in x)
    rows.append({"адаптер": n, "train loss, шаг 2": tr[2], "train loss, шаг 56": tr[56],
                 "eval loss, шаг 4": ev[4], "eval loss, шаг 20": ev[20], "eval loss, шаг 40": ev[40],
                 "eval loss, финал": ev[max(ev)], "время, мин": run["train_runtime"] / 60})
pd.DataFrame(rows).set_index("адаптер").round(3)"""),
md("""Loss падает ровно, без скачков. Резкий спуск идёт первые ~15 шагов (одна эпоха ≈ 19 шагов):
модель выучивает каркас `FETCH [..] WHERE @.. AS ..`. Дальше идёт медленное уточнение значений.

Признаков переобучения нет. Loss на тесте всё время идёт вместе с train loss и до конца не
растёт. Последние ~10 шагов она стоит на месте (0.074 у r8, 0.053 у r16), так что 3 эпохи —
верхняя граница: дальше модель только запоминала бы обучающие примеры. У r=16 обе кривые
ниже на всём пути: больший ранг быстрее и глубже подгоняется под формат."""),

md("""## 4. Тестирование на отложенной выборке

Оба адаптера подключены к **свежезагруженной** базовой модели (`PeftModel.from_pretrained`)
и прогнаны по 36 тестовым запросам, которых модель не видела. Генерация жадная.
Метрики (`src/evaluate.py`):

* **синтаксис:** ответ разбирается грамматикой ECQL (`@`, `[]`, `&&`/`||`, скобки, кавычки);
* **логика:** та же сущность, тот же формат `AS`, те же условия (поле, оператор, значение) и
  та же структура И/ИЛИ. Порядок операндов внутри `&&`/`||` не важен. Отдельно считается
  путаница `ABOVE`/`BELOW` (поле и значение верны, оператор нет);
* **галлюцинации:** в ответе есть элементы SQL (`SELECT`, `FROM`, `AND`, `OR`, `=`, `>`…)."""),
code("""groups = {
    "синтаксис": ["syntax_valid", "schema_valid"],
    "логика": ["entity_correct", "conditions_correct", "format_correct", "logic_correct", "exact_match"],
    "галлюцинации": ["sql_hallucination"],
}
rows = []
for g, keys in groups.items():
    for k in keys:
        rows.append({"критерий": g, "метрика": METRIC_NAMES[k],
                     **{n: f"{metrics[n][k]:.0%} ({round(metrics[n][k] * 36)}/36)" for n in ("r8", "r16")}})
rows.append({"критерий": "логика", "метрика": "перепутан ABOVE/BELOW или IS/NOT (условий)",
             **{n: str(metrics[n]["operator_errors_total"]) for n in ("r8", "r16")}})
pd.DataFrame(rows).set_index(["критерий", "метрика"])"""),
code("""lvl = pd.DataFrame({n: pd.DataFrame(p).groupby("level")["logic_correct"].agg(lambda s: f"{s.sum()}/{len(s)}")
                    for n, p in preds.items()}).loc[order]
lvl.index.name = "логически верно по уровням"
lvl"""),

md("""### 5 примеров из тестовой выборки (адаптер r=16)

По одному запросу каждого уровня сложности плюс один ошибочный ответ."""),
code("""p16 = preds["r16"]
pick = [next(r for r in p16 if r["level"] == l and r["logic_correct"]) for l in ("simple", "and", "or", "nested")]
pick.append(next(r for r in p16 if r["input"].startswith("Активные проекты Смирнова")))
pd.DataFrame([{"вход на русском": r["input"], "эталонный ECQL": r["reference"],
               "ответ модели": r["prediction"], "верно": "✅" if r["logic_correct"] else "❌"} for r in pick])"""),

md("""### Все ошибки

Ни одной строчки SQL, ни одной путаницы `ABOVE`/`BELOW`. Ошибаются модели не в синтаксисе
ECQL, а в «словаре»: какое английское значение и какое поле соответствуют русскому слову."""),
code("""for n in ("r16", "r8"):
    bad = [r for r in preds[n] if not r["logic_correct"]]
    display(Markdown(f"**{n}: {len(bad)} ошибок из 36**"))
    display(pd.DataFrame([{"уровень": r["level"], "вход": r["input"], "эталон": r["reference"],
                           "модель": r["prediction"], "что не так": r["error"]} for r in bad]))"""),

md("""### Разбор ошибок

**1. «Активные проекты Смирнова…» → `@manager IS 'Smirnova'`** (обе модели, и такой же случай
«Сделки Кузнецовой или Смирнова»). «Смирнова» — это и родительный падеж мужской фамилии
(«проекты Смирнова»), и женская фамилия. В обучающей выборке форма «Смирнова» встречается
**один раз**, как женское имя: «Найди сотрудника по имени Анна Смирнова» → `'Anna Smirnova'`.
Модель честно выучила это соответствие. «Иванова» (в train трижды в родительном падеже →
`'Ivanov'`) r16 переводит верно, а r8 и здесь пишет `'Ivanova'`. Причина — **недостаток данных** на неоднозначную
морфологию: нужны примеры «проекты Смирнова → Smirnov» рядом с «Анна Смирнова → Anna Smirnova».
r16 к тому же потерял условие `@status IS 'active'`. Единственное слово «активные» в начале
запроса проглочено, хотя в остальных запросах с ним модель справляется.

**2. «Нужна таблица со всеми сотрудниками» → нет `AS TABLE`** (обе модели). Во всех 14
обучающих примерах с `AS TABLE` формат задан наречием «таблицей» или оборотом «в виде
таблицы». Здесь «таблица» — подлежащее: то, что просят. Модель выучила **поверхностный
шаблон** («таблицей» → `AS TABLE`), а не смысл. Лечится разнообразием формулировок в датасете.

**3. «Кто из бухгалтеров не в отпуске?» → `@on_leave IS false`** (r16, ещё раз в последнем
nested-запросе). Это галлюцинация, но не SQL: модель придумала булево поле, как в JSON или
Python. «Не в отпуске» встречается в train один раз (`@status NOT 'on_leave'`), и
предобученная привычка «флаг = true/false» его перевесила. Парсер отвергает такой ответ:
булевых значений в ECQL нет. Это обе синтаксические ошибки r16.

**4. Редкие поля и значения.** `@item` есть в одном обучающем примере (ThinkPad X1), поэтому
«MacBook Air» уходит в `@category`, а «Офисное кресло» в `@position` (поле сотрудников).
«Феникс» → `'Phenix'` — транслитерация вместо перевода. «Меньше года» → `BELOW 12`: в запросе
нет числа, и модель перевела год в месяцы.

**r8 против r16.** У r8 сверх этого ошибки **структуры**. В двух вложенных запросах он теряет
скобки: `@position IS 'analyst' || @position IS 'engineer' && @city NOT 'Moscow'`. Это
синтаксически корректно, но из-за приоритета `&&` логика другая, отсюда 100% синтаксиса при
худшей логике. Ещё r8 дважды берёт `@status` у `[DEALS]` вместо `@stage` (поле из другой
сущности), упрощает тройную вложенность в сделках по Сибири и чаще переводит значения
«от себя»: «заморожены» → `'frozen'`, «канцелярия» → `'stationery'`, «в отпуске» →
`@vacation_status`."""),

md("""## 5. Выводы

**Удалось ли выучить язык? Да, синтаксис выучен.** После 3 минут обучения на 148 примерах
модель пишет грамматически корректный ECQL в 94–100% случаев. Сущность выбирается верно в
94–100% случаев, `ABOVE`/`BELOW` и `IS`/`NOT` не перепутаны ни разу, SQL не появился ни разу.
Логически точный запрос получается в **72%** случаев (r16; точное совпадение строки — 69%).
Оставшиеся ошибки почти все словарные: перевод значений (родительный падеж фамилий, «Феникс»)
и выбор редкого поля. Это следствие небольшого датасета, а не непонимания языка: ошибки
приходятся на значения и поля, которые встречаются в обучении один-два раза.

**Влияние ранга r.** r=16 лучше r=8 по логической точности (26 против 23 из 36) и по loss
на тесте (0.053 против 0.074).

* **Где разница:** почти вся она во **вложенных** запросах. Здесь r8 верен в 0 из 6, а r16 в
  3 из 6. На простых запросах и запросах с одной связкой ранги почти равны (simple 11/14 у
  обоих, and 8/10 против 7/10, or 3/4 против 4/4). Ранг 8 хватает на «словарь» простых
  запросов, но не на вложенную структуру.
* **Синтаксис ECQL:** r16 точнее в **структуре**: он сохраняет скобки во вложенных запросах и не
  путает поля между сущностями. Зато r16 дважды выдал выдуманное `@on_leave IS false` и
  получил 94% синтаксиса против 100% у r8. Значит, больший ранг лучше передаёт структуру, но
  не убирает пробелы в словаре.
* **Оговорка:** разница — 3 запроса из 36 на одном прогоне каждого ранга. Тенденция
  согласуется с loss, но для твёрдого вывода нужна выборка больше.

**Что улучшило бы результат:** больше разных формулировок формата («таблица», «в виде
списка»), пары «родительный падеж ↔ женская фамилия», больше примеров с редкими полями
(`@item`, `@name`). Синтаксис уже выучен, теперь не хватает данных."""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "llm-7", "display_name": "llm-7", "language": "python"},
               "language_info": {"name": "python"}}
nbf.write(nb, ROOT / "notebooks/report.ipynb")
print("notebooks/report.ipynb")
