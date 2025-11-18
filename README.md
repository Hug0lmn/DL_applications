# DL_applications — Deep Learning for Lyrics Generation

This repository explores deep learning applications in NLP, with a focus on rap/hip-hop lyrics generation.  
It provides complete pipelines for:

- Corpus scraping (Genius API)
- Cleaning and normalization
- Character-level and subword tokenization
- FastText embedding
- Training RNN, LSTM, and Transformer models
- Generation via Streamlit apps
- Simple Markov-based song-structure modeling

The objective is to compare different text granularities (character vs subword) and model families, and analyze their impact on lyrical quality, structure, and coherence.

---

## Repository Structure

DL_applications/  
├── Get_lyrics.py # Corpus scraping (Genius) + cleaning orchestration  
├── Streamlit_char.py # Streamlit app for character-level generation  
├── Streamlit_subword.py # Streamlit app for subword-level generation (WIP)  
│  
├── Corpus/  
│ ├── Cleaning_txt.py # Cleaning, de-duplication, normalization  
│ ├── Manip.py # Corpus manipulation utilities  
│ └── Tokenizer.py # Tokenizer training (BPE), FastText embedding, context-window creation  
│  
├── Models/  
│ ├── ... # Checkpoints for RNN/LSTM/Transformer models  
│ └── Training_example.ipynb # Example training notebook  
│  
├── Markov/  
│ └── Markov_Chain.py # Simple transition matrix for song-part sequencing  
│  
└── Functions/  
└── ... # Helper functions used by the Streamlit apps  


---

## Notes on Current Experiments

- **Transformers at character level underperform** compared to RNNs/LSTMs despite similar perplexity.  
  Expected causes:
  - characters carry very limited semantic information  
  - sequence lengths are much longer  
  - Transformers do not leverage recurrence like RNNs  

- **Subword-level models (BPE ~4000 vocab)** produce better semantic consistency and more coherent lyrical structure.

---

## Next Steps / Research Ideas

### 1. Phoneme-level modeling
Phonemes carry more information than characters and can improve rhyme, flow, and syllabic structure.  
Plan: process the corpus with a phonemizer, then train models on phoneme sequences.

### 2. Experiment with xLSTM
Test the xLSTM architecture (recent research) for better handling of long-range dependencies and improved training stability.

---

## Quick Usage

### 1. Build a corpus from specific artists

The script will:
- prompt for artist/song names  
- download lyrics from Genius  
- clean and normalize the corpus  
- generate a simple Markov transition matrix  

Command:

```bash
python Get_lyrics.py --RNN True --nb <number_of_artists>
```

### 2. Train models

Use the notebook :  
```bash
Models/.../Training_example.ipynb
```  

### 3. Generate lyrics (Streamlit)

Character_level generation :  
```bash
streamlit run Streamlit_char.py  
```  

Subword_level generation (under development) :  
```bash
streamlit run Streamlit_subword.py
```  