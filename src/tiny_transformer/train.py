"""Train on Don Quijote. Run: python -m tiny_transformer.train [--steps 6000]

Compares against a bigram model trained on the same split, so "the loss went down"
has a reference. The bigram is computed by counting and is the best any model can do
when it only sees the previous character.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from credit_risk.data import ARTIFACTS, REPORTS, ROOT

from .model import Config, TinyGPT

SOURCE = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
RAW = ROOT / "data" / "raw" / "quijote.txt"
INK, MUTED, GRID, BLUE, ORANGE = "#1f2937", "#6b7280", "#e5e7eb", "#2a78d6", "#eb6834"


def load_text() -> str:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE, timeout=60) as response:  # nosec B310 - fixed https URL
            RAW.write_bytes(response.read())
    text = RAW.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    start = text.index("*** START OF")
    start = text.index("\n", start) + 1
    return text[start : text.index("*** END OF")].strip()


class Tokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.to_id = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.to_id[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


def get_batch(data: torch.Tensor, block: int, batch: int, gen: torch.Generator):
    starts = torch.randint(len(data) - block - 1, (batch,), generator=gen)
    x = torch.stack([data[s : s + block] for s in starts.tolist()])
    y = torch.stack([data[s + 1 : s + block + 1] for s in starts.tolist()])
    return x, y


def bigram_loss(train: torch.Tensor, val: torch.Tensor, vocab: int) -> float:
    counts = torch.ones(vocab, vocab)  # add-one smoothing
    counts.index_put_((train[:-1], train[1:]), torch.ones(len(train) - 1), accumulate=True)
    log_probs = torch.log(counts / counts.sum(1, keepdim=True))
    return float(-log_probs[val[:-1], val[1:]].mean())


@torch.no_grad()
def estimate_loss(model: TinyGPT, data: torch.Tensor, gen: torch.Generator, n: int = 40) -> float:
    model.eval()
    losses = []
    for _ in range(n):
        x, y = get_batch(data, model.cfg.block_size, 64, gen)
        losses.append(float(model(x, y)[1]))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--block", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(1337)
    gen = torch.Generator().manual_seed(1337)
    text = load_text()
    tokenizer = Tokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    cut = int(0.9 * len(data))  # contiguous split: the end of the novel is never seen
    train, val = data[:cut], data[cut:]

    cfg = Config(vocab_size=len(tokenizer.chars), block_size=args.block)
    model = TinyGPT(cfg)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )
    warmup = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps - warmup),
        ],
        milestones=[warmup],
    )
    n_params = sum(p.numel() for p in model.parameters())
    baseline = bigram_loss(train, val, cfg.vocab_size)
    print(
        f"{len(data):,} chars, vocab {cfg.vocab_size}, {n_params:,} params, bigram {baseline:.3f}"
    )

    history = []
    start = time.time()
    interval = max(1, args.steps // 30)
    for step in range(args.steps + 1):
        if step % interval == 0:
            row = {
                "step": step,
                "train": round(estimate_loss(model, train, gen), 4),
                "val": round(estimate_loss(model, val, gen), 4),
            }
            history.append(row)
            print(row, f"{time.time() - start:.0f}s", flush=True)
        if step == args.steps:
            break
        x, y = get_batch(train, cfg.block_size, args.batch, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    prompt = "En un lugar de la Mancha"
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    sample = tokenizer.decode(model.generate(context, 400)[0].tolist())

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(
        {"state": model.state_dict(), "config": cfg, "chars": tokenizer.chars},
        ARTIFACTS / "tiny_gpt.pt",
    )
    report = {
        "characters": len(data),
        "vocab": cfg.vocab_size,
        "parameters": n_params,
        "steps": args.steps,
        "device": "cpu",
        "train_seconds": round(time.time() - start),
        "bigram_val_nll": round(baseline, 4),
        "final_val_nll": history[-1]["val"],
        "best_val_nll": min(r["val"] for r in history),
        "history": history,
        "sample": sample,
    }
    REPORTS.joinpath("tiny_transformer.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot(history, baseline)
    print(sample)


def plot(history: list[dict], baseline: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    steps = [r["step"] for r in history if r["step"] > 0]
    train = [r["train"] for r in history if r["step"] > 0]
    val = [r["val"] for r in history if r["step"] > 0]
    ax.plot(steps, train, color=BLUE, linewidth=2, label="Train")
    ax.plot(steps, val, color=ORANGE, linewidth=2, label="Validation")
    ax.axhline(baseline, color=MUTED, linewidth=1, linestyle="--")
    ax.text(
        steps[-1], baseline, "bigram baseline", color=MUTED, fontsize=9, ha="right", va="bottom"
    )
    ax.set_xlabel("Training step", color=INK)
    ax.set_ylabel("Cross-entropy (nats per char)", color=INK)
    ax.set_title("TinyGPT on Don Quijote", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, labelcolor=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "transformer_loss.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
