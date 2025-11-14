import streamlit as st
import pickle
import fasttext
from functools import lru_cache
import time
from pathlib import Path

from Functions.Generation.Generation import load_and_clean, generate_from_text
from Functions.Models.Char_models import Char_Models

script_dir = Path(__file__).resolve().parent

path = script_dir / "Corpus" / "Corpora" / "fasttext_embedding"
FASTTEXT = fasttext.load_model(str(path))   

@lru_cache(maxsize=3)
def load_model(choice):
    model = Char_Models(choice)
    file_path = script_dir / "Models" / "Char_lvl" / f"Model_{choice}.pt"
    ckpt = load_and_clean(file_path)
    model.load_state_dict(ckpt["model_state_dict"])
    return model

@lru_cache(maxsize=1)
def load_mapping():
    encode_path = script_dir / "Corpus" / "Corpora" / "Char_level" / "encoding_map.pkl"
    with encode_path.open("rb") as f:
        return pickle.load(f)


def main():

    mapping = load_mapping()
    forbidden_generation = [i for i in range(70, 72)]

    st.title("🎵 Rap Lyric Generator")
    st.text("For optimal generation, keep parameter values")

    seed = st.text_input("Enter a starting phrase:")
    temp = st.slider("Sampling temperature", 0.1, 1.2, 0.7)
    length = st.slider("Generation length", 100, 400, 300)
    top_k = st.slider("Top k character", 5, len(mapping), 40)

    model_names = ["RNN", "GRU", "Transformer"]
    hiddens = [True, True, False]

    # ======================================================
    # GENERATION
    # ======================================================
    if st.button("Generate lyrics"):

        with st.spinner("Generating..."):

            st.markdown("<div id='three-rows'>", unsafe_allow_html=True)
            row_rnn = st.container()
            row_gru = st.container()
            row_trans = st.container()
            st.markdown("</div>", unsafe_allow_html=True)

            models = [load_model(name) for name in model_names]

            texts = []
            for model, hidden in zip(models, hiddens):
                t = generate_from_text(
                    model, seed, length, temp, top_k,
                    mapping, FASTTEXT, forbidden_generation, hidden
                )
                texts.append(t)

            #Display results
            titles = ["RNN", "GRU", "Transformer"]
            rows = [row_rnn, row_gru, row_trans]

            for title, row, text in zip(titles, rows, texts):

                row.markdown(f"<div class='column-box'><h3>{title}</h3></div>",
                             unsafe_allow_html=True)

                buffer = ""
                placeholder = row.empty()

                for char in text:
                    buffer += char.replace("\n", "<br>")
                    placeholder.markdown(
                        f"""
                        <div class='column-box'>
                            <pre>{buffer}</pre>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    time.sleep(0.01)


if __name__ == "__main__":
    main()
