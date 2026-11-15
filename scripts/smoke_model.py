import torch

from zerre import Config
from zerre.model import TinyVoca

config = Config(speakers=("MK", "NK"))
model = TinyVoca(config).eval()
tokens = torch.tensor([[1, 10, 11, 12, 2]])
stress = torch.zeros_like(tokens)
speakers = torch.tensor([0])
lengths = torch.tensor([tokens.shape[1]])
with torch.no_grad():
    try:
        audio, out_lengths, details = model.synthesize(tokens, stress, lengths, speakers)
    except TypeError:
        audio, out_lengths = model.synthesize(tokens, stress, lengths, speakers)
        details = None
print("params:", sum(p.numel() for p in model.parameters()))
print("vocab:", config.vocab_size, "audio samples:", audio.shape, "length:", int(out_lengths[0]))
print("finite:", bool(torch.isfinite(audio).all()))
