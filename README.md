# DSL Query Generator: дообучение Qwen3-4B переводить русский язык в ECQL

Малая модель **Qwen3-4B-Instruct-2507** (4-bit, QLoRA) дообучена LoRA-адаптером переводить
запросы на русском языке во внутренний язык запросов **ECQL** вымышленной компании «E-Corp».
Это домашнее задание курса OTUS «LLM».

```
«Кто из сотрудников не в Москве и работает аналитиком или инженером? Таблицей.»
        ▼  Qwen3-4B + LoRA r=16
FETCH [EMPLOYEES] WHERE @city NOT 'Moscow' && (@position IS 'Analyst' || @position IS 'Engineer') AS TABLE
```

## Ссылки для сдачи

| что | где |
| --- | --- |
| **Отчёт** (все выводы сохранены, ноутбук читается без запуска) | [`notebooks/report.ipynb`](notebooks/report.ipynb) |
| **Обучение в Colab (T4)** | [`notebooks/train_colab.ipynb`](https://colab.research.google.com/github/AlexIntellySoft/llm-7/blob/main/notebooks/train_colab.ipynb) |
| **Веса адаптеров** | [Alex-Popov/qwen3-4b-ecql-lora-r16](https://huggingface.co/Alex-Popov/qwen3-4b-ecql-lora-r16) (основной), [Alex-Popov/qwen3-4b-ecql-lora-r8](https://huggingface.co/Alex-Popov/qwen3-4b-ecql-lora-r8) |
| **Метрики** | [`outputs/metrics.json`](outputs/metrics.json) |
| **Ответы модели на тестовой выборке** | [`outputs/predictions_r16.jsonl`](outputs/predictions_r16.jsonl), [`outputs/predictions_r8.jsonl`](outputs/predictions_r8.jsonl) |
| **График loss** | [`outputs/loss_curve.png`](outputs/loss_curve.png) |
| **Карта «требование → что сделано → где смотреть»** | [`ДЛЯ_ПРОВЕРЯЮЩЕГО.md`](ДЛЯ_ПРОВЕРЯЮЩЕГО.md) |

## Главное

Тестовая выборка — 36 запросов, которых модель не видела при обучении.

| метрика | LoRA r=8 | **LoRA r=16** |
| --- | --- | --- |
| синтаксис ECQL корректен | 100% | 94% |
| поля соответствуют схеме сущности | 89% | 92% |
| верная сущность | 100% | 94% |
| верный формат вывода (`AS …`) | 97% | 92% |
| верный набор условий | 67% | 75% |
| перепутан `ABOVE`/`BELOW` | 0 | 0 |
| элементы SQL в ответе | **0%** | **0%** |
| логически эквивалентен эталону | 64% (23/36) | **72% (26/36)** |
| точное совпадение строки | 64% | 69% |
| eval loss в конце обучения | 0.074 | **0.053** |
| время обучения на T4 (3 эпохи) | ≈3 мин | ≈3 мин |

Логическая эквивалентность по уровням сложности:

| уровень | примеров в тесте | r=8 | r=16 |
| --- | --- | --- | --- |
| без условий (`all`) | 2 | 1/2 | 1/2 |
| одно условие (`simple`) | 14 | 11/14 | 11/14 |
| `&&` (`and`) | 10 | 8/10 | 7/10 |
| `\|\|` (`or`) | 4 | 3/4 | **4/4** |
| вложенные со скобками (`nested`) | 6 | 0/6 | **3/6** |

![loss](outputs/loss_curve.png)

1. **Синтаксис выучен за 3 минуты на 148 примерах.** `@`, `[]`, `&&` расставлены верно,
   `ABOVE`/`BELOW` не перепутаны ни разу, SQL не появился ни разу.
2. **Оставшиеся ошибки словарные, а не синтаксические.** Модель путает перевод значений и выбор
   редких полей (разбор ниже).
3. **Ранг r влияет на вложенную структуру.** r=8 теряет скобки во вложенных запросах (0/6), r=16
   их сохраняет (3/6). На простых запросах ранги почти равны. Зато r=16 дважды выдумал
   булево поле `@on_leave IS false`, отсюда 94% синтаксиса против 100%.
4. **Переобучения нет:** loss на тесте снижается вместе с train loss и выходит на плато к
   концу 3-й эпохи, без скачков.

## 1. Язык ECQL

Правила из задания:

```
FETCH [СУЩНОСТЬ] [WHERE условия] [AS JSON | AS TABLE | AS LIST]
```

* сущность — заглавными буквами в квадратных скобках: `[EMPLOYEES]`;
* поле — с префиксом `@`: `@salary`;
* операторы: `IS` (=), `NOT` (≠), `ABOVE` (>), `BELOW` (<);
* связки: `&&` (И), `||` (ИЛИ);
* строки — в одинарных кавычках на английском, числа — без кавычек и полностью
  («2,5 млн» → `2500000`).

Что пришлось доопределить, потому что в задании этого нет:

* ключевое слово `WHERE` перед условиями, чтобы условия однозначно отделялись от сущности;
* **скобки** для смешения `&&` и `||`: `A && (B || C)`. Без них вложенный запрос неоднозначен;
* формат `AS …` пишется только если пользователь явно его попросил;
* схема полей и словарь значений, чтобы эталоны были однозначными:

| сущность | поля |
| --- | --- |
| `[EMPLOYEES]` | `@name`, `@city`, `@department`, `@position`, `@status`, `@salary`, `@age`, `@experience` (лет) |
| `[PROJECTS]` | `@name`, `@status`, `@priority`, `@manager`, `@client`, `@budget`, `@team_size`, `@year` |
| `[INVENTORY]` | `@item`, `@category`, `@warehouse`, `@status`, `@quantity`, `@price` |
| `[DEALS]` | `@client`, `@stage`, `@region`, `@manager`, `@amount`, `@probability` (%), `@year` |

Полная спецификация со словарём значений — в [`docs/teacher_prompt.md`](docs/teacher_prompt.md).
Формальная грамматика (токенизатор + рекурсивный парсер) — в [`src/ecql.py`](src/ecql.py).

## 2. Датасет

**Модель-учитель — Claude.** Ключ YandexGPT из прошлых заданий к этому моменту истёк. Учителю
дали правила и схему выше (промпт в [`docs/teacher_prompt.md`](docs/teacher_prompt.md)) и попросили
пары «запрос на русском — ECQL» от простых до вложенных, с разнообразными формулировками:
повелительные («выгрузи»), вопросы («кто из…?»), разговорные («ребята из саппорта»), числа словами
и сокращениями («150 тыс.», «2,5 млн»).

**Проверка качества.** Каждый эталон автоматически проверен парсером `src/ecql.py`: грамматика,
допустимость полей для сущности, типы значений, отсутствие SQL. Пары лежат в
[`tools/teacher_pairs.py`](tools/teacher_pairs.py), сборка — [`tools/build_dataset.py`](tools/build_dataset.py).

**Формат JSONL:**

```json
{"instruction": "Переведи запрос на ECQL", "input": "Ребята из саппорта в Самаре, выведи в json", "output": "FETCH [EMPLOYEES] WHERE @department IS 'Support' && @city IS 'Samara' AS JSON", "level": "and", "entity": "EMPLOYEES"}
```

Поля `level` и `entity` служебные: по ним делается стратифицированный split и считаются метрики
по уровням. В обучение идут только `instruction`, `input` и `output`.

**Состав: 184 пары, split 80/20 → 148 train / 36 test** (seed 42, стратификация по уровню × сущности).

| уровень | всего | train | test |
| --- | --- | --- | --- |
| без условий | 11 | 9 | 2 |
| одно условие | 71 | 57 | 14 |
| несколько условий через `&&` | 53 | 43 | 10 |
| условия через `\|\|` | 20 | 16 | 4 |
| вложенные (`&&` + `\|\|` со скобками) | 29 | 23 | 6 |
| **итого** | **184** | **148** | **36** |

| сущность | всего | train | test |
| --- | --- | --- | --- |
| `EMPLOYEES` | 59 | 48 | 11 |
| `PROJECTS` | 45 | 37 | 8 |
| `DEALS` | 41 | 32 | 9 |
| `INVENTORY` | 39 | 31 | 8 |

Покрытие конструкций (число пар, где конструкция встречается): `IS` 134, `&&` 82, `ABOVE` 50,
`||` 49, `BELOW` 38, скобки 29, `NOT` 24, `AS JSON` 20, `AS TABLE` 19, `AS LIST` 13.

Файлы: [`data/ecql_dataset.jsonl`](data/ecql_dataset.jsonl), [`data/train.jsonl`](data/train.jsonl), [`data/test.jsonl`](data/test.jsonl).

## 3. Модель и LoRA

**Базовая модель — [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).**
4 млрд параметров в 4-bit занимают ≈2,5 ГБ и с запасом помещаются в 15 ГБ T4. Модель хорошо
понимает русский, а версия Instruct-2507 отвечает без блока рассуждений `<think>`, поэтому выход
сразу является ECQL-строкой.

**Квантование (bitsandbytes):** NF4 + double quantization, вычисления в fp16 (T4 не поддерживает bf16).

**LoRA (PEFT):**

| параметр | значение | почему |
| --- | --- | --- |
| `r` | **16** (контроль: 8) | задача узкая (одна грамматика, ~40 полей), высокий ранг не нужен; 16 — стандартная середина, 8 — чтобы проверить, хватит ли меньшего |
| `lora_alpha` | 32 (для r=8 — 16) | alpha = 2r: масштаб обновления одинаков для обоих рангов, сравнение честное |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj` | задача про то, на какие токены запроса смотреть (поле, оператор, значение); attention-проекции — классический выбор QLoRA |
| `lora_dropout` | 0.05 | лёгкая регуляризация на маленьком датасете |
| обучаемых параметров | ≈11,8 млн при r=16 (≈0,3% модели), ≈5,9 млн при r=8 | |

## 4. Обучение

`trl.SFTTrainer` ([`src/finetune.py`](src/finetune.py)):

| параметр | значение |
| --- | --- |
| learning rate | 2e-4, cosine, 3 шага warmup |
| эпохи | 3 (57 шагов оптимизатора) |
| батч | 2 × gradient accumulation 4 = эффективный 8 |
| точность | fp16 + gradient checkpointing |
| max_length | 256 токенов |
| loss | только по ответу (формат `prompt`/`completion`): русский запрос модели и так дан, учить его генерировать незачем |
| eval | на test каждые 4 шага, чтобы видеть переобучение |

Итоговый train loss (среднее за обучение) 0.53, финальный eval loss 0.053 (r=16) и 0.074 (r=8).
Логи по шагам — `outputs/log_history_r16.json`, `outputs/log_history_r8.json`.

**Подводный камень, который пришлось обойти.** Для 4-bit модели `SFTTrainer` в своём `__init__`
переводит веса LoRA в bfloat16 (рекомендация QLoRA для A100/H100). На T4 fp16-обучение через
GradScaler падает с `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not
implemented for 'BFloat16'`. Приведение весов к fp32 *до* создания тренера не помогает: тренер
перезаписывает его. Решение — вернуть обучаемые веса в fp32 **после** создания `SFTTrainer`,
до `trainer.train()`.

## 5. Оценка на отложенной выборке

Адаптеры подключаются к свежей 4-bit базе через `PeftModel.from_pretrained`, генерация жадная
(`do_sample=False`), берётся первая строка ответа. Каждый ответ проверяется скриптом
[`src/evaluate.py`](src/evaluate.py):

* **синтаксис** — ответ разбирается парсером ECQL без ошибок;
* **схема** — все поля существуют у этой сущности;
* **логика** — ответ и эталон приводятся к каноническому дереву условий (порядок внутри `&&`/`||`
  не важен), сравниваются сущность, формат и дерево. Отдельно считается, сколько раз перепутаны
  `ABOVE`/`BELOW`;
* **галлюцинации SQL** — поиск `SELECT`, `FROM`, `=`, `>`, `AND`, `OR` и т.п.

### 5 примеров: вход — эталон — ответ модели (r=16)

| запрос | эталон | ответ модели | |
| --- | --- | --- | --- |
| Проекты Иванова или Петровой с бюджетом выше 3 млн, кроме отменённых | `FETCH [PROJECTS] WHERE (@manager IS 'Ivanov' \|\| @manager IS 'Petrova') && @budget ABOVE 3000000 && @status NOT 'cancelled'` | то же самое | ✅ |
| Сделки в Сибири, которые либо выиграны, либо на переговорах с вероятностью выше 60%, в JSON | `FETCH [DEALS] WHERE @region IS 'Siberia' && (@stage IS 'won' \|\| (@stage IS 'negotiation' && @probability ABOVE 60)) AS JSON` | `… && (@stage IS 'won' \|\| @stage IS 'negotiation' && @probability ABOVE 60) AS JSON` | ✅ (`&&` сильнее `\|\|`, смысл тот же) |
| Товары не из канцелярии с ценой ниже 1000 или количеством выше 500, таблицей | `FETCH [INVENTORY] WHERE @category NOT 'office_supplies' && (@price BELOW 1000 \|\| @quantity ABOVE 500) AS TABLE` | то же самое | ✅ |
| Активные проекты Смирнова, кроме низкоприоритетных | `FETCH [PROJECTS] WHERE @status IS 'active' && @manager IS 'Smirnov' && @priority NOT 'low'` | `FETCH [PROJECTS] WHERE @manager IS 'Smirnova' && @priority NOT 'low'` | ❌ |
| Нужна таблица со всеми сотрудниками | `FETCH [EMPLOYEES] AS TABLE` | `FETCH [EMPLOYEES]` | ❌ |

### Разбор ошибок

1. **«Смирнова» → `'Smirnova'`.** «Проекты Смирнова» — родительный падеж от «Смирнов». Но в
   обучающих данных «Смирнова» встречается только как женское имя (`'Anna Smirnova'`), и модель
   выучила именно это соответствие. Падежи фамилий нужно показывать в датасете явно.
2. **«Нужна таблица…» без `AS TABLE`.** В обучении формат всегда задавался словами «таблицей» или
   «в виде таблицы», а существительное «таблица» в начале фразы модель не связала с форматом.
3. **Выдуманное поле `@on_leave IS false` (r=16).** Вместо `@status NOT 'on_leave'` модель
   придумала булево поле — значение статуса превратилось в имя поля. Из-за этого 2 ответа не
   проходят парсер (94% синтаксиса).
4. **Редкие поля.** `@item` есть всего в одном обучающем примере, и «MacBook Air» модель кладёт
   в `@category`, а «Офисное кресло» — в несуществующий `@position`. Сюда же «Феникс» →
   `'Phenix'` и «меньше года» → `@experience BELOW 12` (месяцы вместо лет).
5. **Только у r=8:** теряет скобки во вложенных запросах, пишет `@status` вместо `@stage` для
   сделок и сам переводит значения (`'frozen'` вместо `'on_hold'`, `'stationery'` вместо
   `'office_supplies'`).

## 6. Выводы

* **Язык выучен.** Грамматика ECQL усвоена полностью: операторы не путаются, SQL нет ни в одном
  ответе, сущность определяется практически всегда. Для этого хватило 148 примеров и 3 минут на T4.
* **Ошибки идут от данных, а не от модели.** Почти все промахи — значения и поля, которые в
  обучении встречались 0–1 раз или в другой форме. Лечится расширением датасета (падежи фамилий,
  больше `@item`, разные способы попросить формат), а не изменением гиперпараметров.
* **Влияние r.** r=16 лучше r=8 на 8 п.п. по логике (72% против 64%), и почти весь выигрыш — во
  вложенных запросах (3/6 против 0/6): удвоенной ёмкости адаптера хватает, чтобы выучить
  расстановку скобок. На простых запросах ранги равны. Цена r=16 — чуть больше «фантазии»
  (выдуманное `@on_leave`). Для этой задачи выбран **r=16**.

## Структура проекта

```
llm-7/
├── data/                    датасет: ecql_dataset.jsonl, train.jsonl (148), test.jsonl (36)
├── docs/teacher_prompt.md   правила ECQL, схема, промпт модели-учителю
├── src/
│   ├── ecql.py              грамматика ECQL: токенизатор, парсер, схема, канонизация
│   ├── evaluate.py          метрики: синтаксис, схема, логика, SQL
│   └── finetune.py          4-bit загрузка, LoRA, SFTTrainer, генерация
├── tools/
│   ├── teacher_pairs.py     184 пары от модели-учителя
│   ├── build_dataset.py     проверка пар парсером и split 80/20
│   ├── smoke_test.py        весь пайплайн на крошечной модели на CPU
│   └── make_report.py       собирает notebooks/report.ipynb
├── notebooks/
│   ├── train_colab.ipynb    обучение и оценка в Colab (T4)
│   └── report.ipynb         отчёт
├── outputs/                 метрики, ответы модели, график loss, логи обучения
└── ДЛЯ_ПРОВЕРЯЮЩЕГО.md      карта требований задания
```

```
docs/teacher_prompt.md (правила ECQL + схема полей)
   │  модель-учитель (Claude) → tools/teacher_pairs.py, 184 пары
   ▼
tools/build_dataset.py — каждый эталон проверен парсером src/ecql.py, split 80/20
   ▼
data/train.jsonl (148), data/test.jsonl (36)    {"instruction", "input", "output"}
   │  notebooks/train_colab.ipynb на T4:
   │  src/finetune.py — 4-bit NF4 (bitsandbytes) → LoRA q/k/v/o (PEFT) → trl.SFTTrainer
   ▼
адаптеры r=8, r=16 → Hugging Face Hub; генерация на тесте → src/evaluate.py → outputs/
   ▼
notebooks/report.ipynb (tools/make_report.py) — отчёт по outputs/, без GPU
```

## Запуск

**Обучение** (нужна CUDA, поэтому Colab):

1. Открыть [`train_colab.ipynb` в Colab](https://colab.research.google.com/github/AlexIntellySoft/llm-7/blob/main/notebooks/train_colab.ipynb).
2. Runtime → Change runtime type → **T4 GPU**.
3. Добавить токен Hugging Face с правом записи в Secrets (значок ключа) под именем `HF_TOKEN`.
4. Runtime → Run all. Оба адаптера обучаются ≈7 минут, плюс скачивание базовой модели.
   В конце адаптеры загружаются на Hugging Face, а `llm7_results.zip` с метриками и ответами
   модели скачивается на компьютер.

**Локально** (датасет, отчёт и проверка пайплайна на крошечной модели, GPU не нужен):

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m tools.build_dataset        # проверка пар и split
.venv/bin/python -m tools.smoke_test           # весь пайплайн на tiny-Qwen3, CPU
.venv/bin/python -m tools.make_report && .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb
```

**Использование адаптера:**

```python
from peft import PeftModel
from src.finetune import load_base, generate

base, tok = load_base()  # Qwen3-4B-Instruct-2507 в 4-bit
model = PeftModel.from_pretrained(base, "Alex-Popov/qwen3-4b-ecql-lora-r16")
print(generate(model, tok, [{"instruction": "Переведи запрос на ECQL",
                             "input": "Сделки на Урале дороже 2 млн, в JSON"}]))
```

## Версии

transformers 5.22, peft 0.21.2, trl 1.16, bitsandbytes 0.50.2 — полный список в [`requirements.txt`](requirements.txt).
