# DL_applications

A research / demo repository exploring deep learning applications in NLP, focused on lyrics generation.  
Implements end-to-end pipelines for tokenization, training and generation using RNN, LSTM and Transformer models at multiple granularities (character, subword). Includes utilities for corpus scraping, cleaning and simple song-structure generation (Markov).

## Repository layout
- Corpus/Preprocessing
  - Get_lyrics.py — scraping (Genius)
  - Cleaning_txt.py, Manip.py — cleaning and normalization
- Tokenization & Embeddings
  - Tokenize.ipynb — tokenizer creation, FastText / custom embeddings
- Models
  - Checkpoints and model code for RNN / LSTM / Transformer (character & subword levels)
- Generation
  - Notebooks and scripts for priming/generation (rep_penalty, top_k)
- Markov
  - Markov_Chain.py — simple song-part transition matrices

## Project status & progress
- Scraping & preprocessing: ✅ implemented (Get_lyrics.py, Cleaning_txt.py)  
- Tokenization & embeddings: ✅ implemented (Tokenize.ipynb)  
- Song-structure (Markov): ✅ transition matrix generation (Markov_Chain.py)  
- Character-level models: ✅ RNN, LSTM, Transformer (training notebooks in Granularity/Character_lvl/Training)  
- Subword-level models: ✅ RNN, LSTM, Transformer (training notebooks in Granularity/Subword_lvl/Training)  

Note: In current experiments Transformers underperform compared to LSTMs/RNNs — likely due to dataset size; see "Next steps" below.

## Next steps / ideas
- Phonemize corpus to improve rhyme/flow modeling.  
- Data augmentation to help Transformer training.  
- Add Luong attention to LSTM.  
- Experiment with xLSTM (research reference).

## Quick usage
1. Scrape and prepare data:
   - python Get_lyrics.py --RNN True --nb <number_of_artists>
   - The script prompts for artist/song names, builds the corpus and runs Markov_Chain.py and cleaning scripts.
2. Train models:
   - See training notebooks under Granularity/*/Training/.
3. Generate text:
   - Use notebooks under Granularity/*/Generation/. They include priming, top_k, repetition penalty.
