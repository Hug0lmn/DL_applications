# DL_applications

This repository serves as a way to practice concepts learned in the Stanford NLP Course.

## Roadmap
- **Scrape** all the song lyrics of a specified artist to create a training corpus (`Get_lyrics.py`).
- **Embed** words from the corpus into an N-dimensional vector space (`Tokenize.ipynb`).
- **Character generation**:
  - Building and using an RNN. In progress (`RNN_char_generation.ipynb`)
  - Building and using an LLM.
- **Word generation**:
  - Building and using an RNN or LLM.
  - Using FastText pretrained embeddings.
- **Sequence generation** models (TBD).

---

## Get_lyrics.py
This script:
- Scrapes an artist’s lyrics from Genius.
- Cleans and formats them.
- Generates two corpora:  
  1. **RNN corpus** – keeps line breaks for sequence models.  
  2. **Tokenization corpus** – cleaned for embedding models like FastText.

### Usage
Run the script directly to fetch and prepare the data:

```bash
python Get_lyrics.py --link https://genius.com/artists/Limsa-daulnay https://genius.com/artists/Isha
