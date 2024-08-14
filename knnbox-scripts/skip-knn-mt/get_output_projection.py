import torch
from fairseq.models.transformer import TransformerModel

model = TransformerModel.from_pretrained(
    "../../pretrain-models/wmt19.de-en/",
    "wmt19.de-en.ffn8192.pt"
)
torch.save({'weight': model.state_dict()['models.0.decoder.output_projection.weight']},
           '../../pretrain-models/wmt19.de-en/output_projection.pt')