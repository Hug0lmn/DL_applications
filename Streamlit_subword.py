import streamlit as st
import pickle
import fasttext
from functools import lru_cache
from transformers import PreTrainedTokenizerFast
from pathlib import Path

from Functions.Generation.Generation import load_and_clean, generate_from_text
from Functions.Models.Subword_models import Subword_Models

script_dir = Path(__file__).resolve().parent

path = script_dir / "Corpus" / "Corpora" / "fasttext_embedding"
FASTTEXT = fasttext.load_model(str(path))   

@lru_cache(maxsize=3)
def load_model(choice):
    model = Subword_Models(choice)
    file_path = script_dir / "Models" / "Subword_lvl" / f"Model_{choice}.pt"
    ckpt = load_and_clean(file_path)
    model.load_state_dict(ckpt["model_state_dict"])
    return model

@lru_cache(maxsize=1)
def load_mapping():
    token_path = script_dir / "Corpus" / "Corpora" / "Subword" / "rap_tokenizer.json"
    return PreTrainedTokenizerFast(
        tokenizer_file = str(token_path),
        pad_token = "<PAD>"
    )

def main():

    mapping = load_mapping()
#    forbidden_generation = [i for i in range(70, 72)]
    forbidden_ids = [i for i in range(2,8)]

    id_not_penalized = [211, 26, 43, 1474, 24, 567]
    forbidden_ids = [i for i in range(2,8)]
    end_of_gen = [i for i in range(8,13)]+[27]

    st.title("🎵 Rap Lyric Generator using subword lvl model")
    st.text("For optimal generation, keep parameter values")

    seed = st.text_input("Enter a starting phrase:")
    temp = st.slider("Sampling temperature", 0.1, 1.2, 0.7)
    length = st.slider("Generation length", 100, 400, 300)
    top_k = st.slider("Top k character", 5, len(mapping), 40)

    model_names = ["RNN", "LSTM", "Transformer"]
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
