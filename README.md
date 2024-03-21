# Image Similarity using CLIP model

Send `POST /compare` with the following body:
```json
{
    "texts": ["shirtless old man"],
    "images": ["base64-of-your-image"]
}
```

to get the similarity score:

```json
[0.9075256707526943]
```

It also supports batching. Just send more than one text in the array. Positions of text and images in respective arrays should match.

Here's an example curl:
```bash
curl 'http://example.com:5680/compare' \
--header 'Content-Type: application/json' \
--data '{
    "texts": ["shirtless old man"],
    "images": ["base64-of-your-image"]
}'
```
