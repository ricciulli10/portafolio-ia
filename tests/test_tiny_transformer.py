import pytest

torch = pytest.importorskip("torch")

from tiny_transformer.model import Config, TinyGPT  # noqa: E402
from tiny_transformer.train import Tokenizer, bigram_loss  # noqa: E402

pytestmark = pytest.mark.torch


def small(**kwargs) -> TinyGPT:
    cfg = Config(
        vocab_size=11, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0, **kwargs
    )
    return TinyGPT(cfg)


def test_output_shape():
    logits, loss = small()(torch.zeros(3, 8, dtype=torch.long))
    assert logits.shape == (3, 8, 11) and loss is None


def test_attention_is_causal():
    """Changing a later token must not affect the prediction for any earlier position."""
    model = small().eval()
    a = torch.randint(0, 11, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 11
    assert torch.allclose(model(a)[0][:, :8], model(b)[0][:, :8])
    assert not torch.allclose(model(a)[0][:, 8:], model(b)[0][:, 8:])


def test_initial_loss_matches_uniform():
    torch.manual_seed(0)
    x = torch.randint(0, 11, (8, 12))
    _, loss = small()(x, x)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(11.0)).item(), abs=0.4)


def test_overfits_one_batch():
    torch.manual_seed(0)
    model = small()
    x = torch.randint(0, 11, (4, 12))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(250):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_rejects_too_long_input():
    with pytest.raises(ValueError):
        small()(torch.zeros(1, 17, dtype=torch.long))


def test_generate_appends_tokens():
    out = small().generate(torch.zeros(1, 3, dtype=torch.long), steps=5)
    assert out.shape == (1, 8)


def test_tokenizer_roundtrip():
    tok = Tokenizer("hola ñandú")
    assert tok.decode(tok.encode("ñandú")) == "ñandú"


def test_bigram_learns_a_deterministic_sequence():
    seq = torch.tensor([0, 1] * 500)
    assert bigram_loss(seq, seq, 2) < 0.01
