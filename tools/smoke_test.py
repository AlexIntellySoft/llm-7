"""Проверка кода обучения и оценки до запуска в Colab: крошечная модель, CPU, без 4-bit.

Модель случайная, поэтому ответы — мусор; проверяется только то, что весь путь
(датасет → SFTTrainer → сохранение адаптера → подключение → генерация → метрики) работает.

    .venv/bin/python -m tools.smoke_test
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import score_one, summarize  # noqa: E402
from src.finetune import attach_adapters, generate, load_base, load_jsonl, loss_points, train_adapter  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"

train = load_jsonl(ROOT / "data/train.jsonl")[:16]
test = load_jsonl(ROOT / "data/test.jsonl")[:4]
with tempfile.TemporaryDirectory() as tmp:
    adapters = {}
    for r in (8, 16):
        out = f"{tmp}/r{r}"
        hist = train_adapter(train, test, r=r, lora_alpha=2 * r, out_dir=out,
                             model_id=TINY, quantize=False, max_steps=4)
        print(f"r={r}: точек train loss {len(loss_points(hist)[1])}, eval loss {len(loss_points(hist)[3])}")
        adapters[f"r{r}"] = out
    base, tok = load_base(TINY, quantize=False)
    model = attach_adapters(base, adapters)
    for name in adapters:
        model.set_adapter(name)
        preds = generate(model, tok, test, batch_size=2, max_new_tokens=8)
        print(name, summarize([score_one(t["output"], p) for t, p in zip(test, preds)]))
print("OK")
