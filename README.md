# DL_applications

This repository explores deep learning applications in NLP, inspired by concepts from the Stanford NLP Course.
The focus is on building end-to-end pipelines for lyrics-based text generation using RNNs, LSTMs, and embeddings.

## Roadmap
**Scraping & Preprocessing**   
✅ Scrape lyrics from Genius [Get_lyrics.py]  
✅ Clean and prepare corpora for RNNs & embeddings [Cleaning_txt.py]  

**Embedding & Tokenization** [Tokenize.ipynb]  
✅ Embed words into N-dimensional vector space (FastText, custom)  

**Structure Generation**  
✅ Transition probabilities between song parts (Markov Chains) [Markov_Chain.py]  
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

## Setup
Install the required dependencies before running the project:

```bash
pip install -r requirements.txt
```

## Usage
Run the script directly to fetch and prepare the data:

python Get_lyrics.py --link https://genius.com/artists/Isha

This command will:
1. Run Get_lyrics.py to generate a corpus of songs and lyrics from the specified artist.  
2. Run Markov_Chain.py to process the data and return the transition matrix.  
3. Run Cleaning_txt.py to clean the corpus by removing or replacing invalid characters and unwanted parts.