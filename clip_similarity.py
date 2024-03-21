import base64
import io
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify
import torch
from rake_spacy import Rake
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine
from PIL import Image
import spacy

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('device:', device)
# model_name = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
model_name = "openai/clip-vit-large-patch14"
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name).to(device)

nlp = spacy.load("en_core_web_trf")
rake = Rake(nlp=nlp)

app = Flask(__name__)

KEYWORD_IMPORTANCE_THRESHOLD = 2.0

# specific to descriptions produced by llava1.6
blacklist = [
    'suggest',
    'accentuate',
    'emphasize',
    'highlight',
    'add',
    'show',
    'indicate',
    'give',
    'demonstrate',
    'hint',
    'signify',
    'likely',
    'probably',
    'imply',
    'exude',
    'like',
    'despite',
    'allow',
    'include',
]

solo_blacklist = [
    'wear',
    'appearance',
    'attire',
    'eye',
    'straight',
    'rest',
]


def is_good_keyword(keyword, local_blacklist):
    tokens = nlp(keyword)
    if len(tokens) == 1:
        if tokens[0].lemma_ in solo_blacklist or tokens[0].text in solo_blacklist:
            return False

    has_noun = False
    for token in tokens:
        if token.lemma_ in blacklist:
            return False
        if token.text in local_blacklist:
            return False
        if token.pos_ == "NOUN":
            has_noun = True

    return has_noun


def extract_keywords(text):
    doc = nlp(text)
    persons = set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])

    keywords = rake.extract_keywords_from_text(text)
    keys: list[str] = [span.text for (weight, span) in keywords
                       if weight >= KEYWORD_IMPORTANCE_THRESHOLD and span.text not in persons]

    keys = list(set(keys))
    good_keys = []
    for key in keys:
        if is_good_keyword(key, persons):
            good_keys.append(key)

    return ", ".join(good_keys)


@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()

    images = [Image.open(io.BytesIO(base64.b64decode(img))) for img in data['images']]
    texts = data['texts']

    with ThreadPoolExecutor() as executor:
        keywords = list(executor.map(extract_keywords, texts))

    image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
    text_inputs = clip_processor(keywords, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = clip_model(pixel_values=image_inputs.pixel_values, input_ids=text_inputs.input_ids)

    image_features = outputs.image_embeds.cpu().detach().numpy()
    keyword_features = outputs.text_embeds.cpu().detach().numpy()

    similarities = []
    for img, kw, keyword in zip(image_features, keyword_features, keywords):
        if not keyword:
            similarities.append(None)
        else:
            similarity = cosine(img, kw)
            similarities.append(similarity)

    return jsonify(similarities)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5680)
