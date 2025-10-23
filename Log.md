# Development Log

As of now, this file serves as a changelog and reasoning log for all modifications, experiments, and upcoming plans.

---

## 🗓️ 23 Oct
### ✅ Changes
- Added multiprocessing for artist lyric scraping.  
- Moved the logic for discarding overly long songs from Markov_Chain.py to Get_lyrics.py.  
- Added new characters to clean/replace in Cleaning_txt.py.  
- Fixed an issue where songs listing the artist as producer were mistakenly included.
- Add new rappers into the corpus

---

### 💡 Next Ideas
As I’m not fully satisfied with the current model performance, here are the possible next directions:

#### 1. Reduce Redundancy
- Identify and remove repetitive parts or sentences beyond an arbitrary limit.  
- Need to define a heuristic or rule for “excess repetition” to avoid overfitting on repeated text.

#### 2. Change Batching and Training Logic
- Group sequences by part type (e.g., <REFRAIN>, <COUPLET>).  
- Rationale: models struggle with long-term dependencies — providing the part type as an auxiliary input could improve intra-part consistency.  
- Possible downside: weaker inter-part coherence.

#### 3. Integrate Phoneme Representations
Phonemes are denser and more information-rich than characters.
- Option 1: Train a standalone phoneme-level model, then convert back to text.  
- Option 2: Use phonemes as additional input features alongside characters.  
- Option 3: Experiment with part-specific phoneme embeddings or conditioning.  
