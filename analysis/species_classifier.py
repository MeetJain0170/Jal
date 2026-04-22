import torch
import numpy as np
from PIL import Image
import open_clip

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_preprocess = None
_tokenizer = None

SPECIES_CLASSES = [
    "clownfish","reef fish","tuna","parrotfish","butterfly fish",
    "lionfish","moray eel","octopus","squid","jellyfish",
    "sea turtle","green sea turtle","hawksbill turtle",
    "shark","great white shark","hammerhead shark","reef shark",
    "whale","blue whale","dolphin","bottlenose dolphin",
    "stingray","manta ray","crab","lobster","coral"
]

def _load():
    global _model,_preprocess,_tokenizer
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        _model.to(_device)
        _model.eval()
        _tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return _model,_preprocess,_tokenizer


def classify_species(img_np, topk=3):
    try:
        model, preprocess, tokenizer = _load()

        image = Image.fromarray(img_np)
        image_input = preprocess(image).unsqueeze(0).to(_device)

        text_tokens = tokenizer(SPECIES_CLASSES).to(_device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).softmax(dim=-1)

        probs = similarity[0].cpu().numpy()
        idx = np.argsort(-probs)[:topk]

        return [
            f"{SPECIES_CLASSES[i]} ({int(probs[i]*100)}%)"
            for i in idx
        ]

    except Exception:
        return []