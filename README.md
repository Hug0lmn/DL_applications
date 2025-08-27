# DL_applications

This repository explores deep learning applications in NLP, inspired by concepts from the Stanford NLP Course.
The focus is on building end-to-end pipelines for lyrics-based text generation using RNNs, LSTMs, and embeddings.

## Roadmap
**Scraping & Preprocessing**   
✅ Scrape lyrics from Genius [Get_lyrics.py]  
✅ Clean and prepare corpora for RNNs & embeddings [Cleaning_txt_Medine.ipynb]  

**Embedding & Tokenization** [Tokenize.ipynb]  
✅ Embed words into N-dimensional vector space (FastText, custom)  

**Structure Generation**  
🔄 Transition probabilities between song parts (Markov Chains) [Markov_Chain.ipynb]  
🔄 Song length distribution analysis  
🔄 Rhyme scheme detection  

**Character Generation**  
🔄 RNN character-level generator [RNN_char_generation.ipynb]  
🔄 LSTM character-level generator [lstm-char-generation.ipynb]  

**Word Generation**  
🔄 RNN/LLM-based word generation  
🔄 FastText pretrained embeddings integration  

**Sequence Generation Models**  
🔜 TBD  

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
