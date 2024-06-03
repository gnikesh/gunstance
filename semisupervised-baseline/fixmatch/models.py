
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

def freeze_layers(model, exclude=[]):
    for param in model.parameters():
        param.requires_grad = False
    for name in exclude:
        for param in getattr(model, name).parameters():
            param.requires_grad = True
    print(f'Num trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'Num total parameters: {sum(p.numel() for p in model.parameters())}')


class HfClf(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.clf = AutoModelForSequenceClassification.from_pretrained(
            args.hf_model,
            num_labels=args.n_classes, 
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        # freeze_layers(self.clf, exclude=['classifier'])

    def forward(self, *args, **kwargs):
        return self.clf(*args, **kwargs)


MODELS = {
    "hf": HfClf,
}


def get_model(args):
    return MODELS[args.model](args)
