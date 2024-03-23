import base64
import io
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from rake_spacy import Rake
from torch import Tensor
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from PIL import Image
import spacy

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('device:', device)
# model_name = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
clip_model_name = "openai/clip-vit-large-patch14"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
embedding_tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
embedding_model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral').to(device)

nlp = spacy.load("en_core_web_trf")
rake = Rake(nlp=nlp)
app = Flask(__name__)

KEYWORD_IMPORTANCE_THRESHOLD = 2.0

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
        # check if lemma of token is in blacklist
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


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def extract_keywords_parallel(prompt):
    return extract_keywords(prompt)


@app.route('/similarity', methods=['POST'])
def evaluate_similarity():
    data = request.get_json()
    prompts = data['prompts']
    descriptions = data['descriptions']

    if len(prompts) != len(descriptions):
        return jsonify({'error': 'The number of prompts and descriptions must be equal.'}), 400

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(extract_keywords_parallel, description) for description in descriptions]
        keywords_list = [future.result() for future in futures]

    batch_size = 32

    similarity_scores = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_keywords = keywords_list[i:i+batch_size]

        # errything goes into single batch
        batch_inputs = batch_prompts + batch_keywords

        batch_dict = embedding_tokenizer(batch_inputs, max_length=4096, padding=True, truncation=True, return_tensors="pt")
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = embedding_model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        embeddings = F.normalize(embeddings, p=2, dim=1)

        prompt_embeddings = embeddings[:len(batch_prompts)]
        keyword_embeddings = embeddings[len(batch_prompts):]

        scores = (prompt_embeddings @ keyword_embeddings.t()) * 100
        similarity_scores.extend(scores.diag().tolist())

    return jsonify(similarity_scores)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5680)
