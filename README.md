# DL_applications
This repository will serve as a way to exercise what I learned through the Stanford NLP Course

It includes a Python script (Get_lyrics.py) that:
- Scrapes an artist’s lyrics from Genius.
- Cleans and formats them.
- Generates two corpora:  
  1. **RNN corpus** (keeps line breaks for sequence models).  
  2. **Tokenization corpus** (cleaned for embedding models like FastText).

Run the script directly to fetch and prepare the data.
Example : python Get_lyrics.py --link https://genius.com/artists/Limsa-daulnay https://genius.com/artists/Isha