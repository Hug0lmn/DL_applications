import argparse
from bs4 import BeautifulSoup as bs
import gc
import multiprocessing
import re
import requests
import sys
import subprocess
from tqdm import tqdm
import unicodedata
import warnings

def search_artist(query, headers): #Will send back 10 results 
    url = f"https://api.genius.com/search"
    params = {"q": query}
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def collect_all_songs(artist_id, page_nb, headers): #Will send back 20 songs of the artist_id
    url = f"https://api.genius.com/artists/{artist_id}/songs?per_page=20&page={page_nb}"
    response = requests.get(url, headers=headers)
    return response.json()

def get_the_artist_id(headers) :

    result_search = search_artist(input("\nName a song of your artist or the artist's name : "), headers)

    #Will try to find the artist specific id in order to find their songs
    for song in result_search["response"]["hits"] : #Loop over all the 10 proposed songs by search_artist()
        
        collect_artist = [] #Collect all the artists on the song
        for i in song["result"]["primary_artists"] : 
            collect_artist.append([i["name"],i["id"]])
        for i in song["result"]["featured_artists"] : 
            collect_artist.append([i["name"],i["id"]])
    
        question = [] 
        for nb,i in enumerate(collect_artist) :
            question.extend([f"{nb} : {i[0]}"])

        #Identify the good artist
        result = input(f"Which artist is the good one (if none leave blank)? {question} : ")
        if result != "" :
            break
    try : 
        result = int(result)
    except :
        warnings.warn("Change the song's name")
        return None
    
    artist_name = collect_artist[result][0]
    artist_id = collect_artist[result][1]
    return artist_name, artist_id

def new_find_discography(artist_name, artist_id) :

    ## Collect the artist's discography 
    collect_songs = []
    
    for i in range(1,50) :
        songs_ = collect_all_songs(artist_id, i, headers)

        if len(songs_["response"]["songs"]) == 0 : #No songs returned
            break
        collect_songs.extend(songs_["response"]["songs"])

    #Check if lyrics are good
    list_songs =[]
    for i in collect_songs : 
        if i["lyrics_state"] == "complete" : #Check that lyrics are complete
            
            #Some rappers are also producers and these songs are included
            matched = False
            for elem in i["primary_artists"] :
                if str(artist_id) in elem["api_path"] :
                    list_songs.append([i["url"],i["title"]])
                    matched = True
                    break

            if not matched : 
                for elem in i["featured_artists"] :
                    if str(artist_id) in elem["api_path"] :
                        list_songs.append([i["url"],i["title"]])
                        matched = True
                        break

    return list_songs

#Fonction GPT
def remove_accents_and_quotes(text):
    # Normalize Unicode and remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Replace curly apostrophes/quotes with straight apostrophe
    text = text.replace("'", "’").replace("'", "‘")
    return text

def Get_lyrics_genius(link, artist_name, session):
    """
    Fetch and clean lyrics from a Genius song page.

    Retrieves the page HTML with `requests`, extracts lyric blocks via BeautifulSoup,
    and returns cleaned lines with normalized spacing and punctuation. Handles
    collaborations ("Featuring") and merges split sections or sentences.

    Args:
        link (str): Genius song URL.
        artist_name (str): Target artist’s name.
        session (requests.Session): Active HTTP session.

    Returns:
        tuple[list[str], bool, requests.Session]:
            Cleaned lyrics, collaboration flag, and session.
    """

    src = session.get(link).text
    new = bs(src,"html.parser")
    artist_name = remove_accents_and_quotes(artist_name)

    # On récupère tous les blocs qui contiennent les paroles
    lyrics_blocks = new.find_all("div", class_="Lyrics__Container-sc-cbcfa1dc-1 dfzvqs") #The class needs to be changed when Genius modify them
    lyrics = []
    for block in lyrics_blocks :
        #If text preview, will be included in text if no action
        if len(block.find_all("a", class_="StyledLink-sc-15c685a-0 iHsqUq SongBioPreview__Wrapper-sc-d13d64be-1 evQSSo"))>0 : #Preview text
            header = block.find("div", class_="LyricsHeader__Container-sc-5b7f377c-1 bMHdTY")
            for i in header.next_siblings :
                if i.get_text(separator="\n") != "":
                    lyrics.append(i.get_text(separator="\n"))
            continue
        if block.get_text(separator="\n") == "":
            continue
        lyrics.append(block.get_text(separator="\n"))
    
    lyrics = "\n".join(lyrics).split("\n")
    paroles = []
    
    collab = False #Indication if the music is a featuring, if so the code will need to check if each part was written by our author
    inside_bracket = False 
    parole_bracket = False

    #Identify a collab (used for post processing and specific scrapping treatment)
    if new.find_all("span", string = "Featuring") :
        collab = True
    else : 
        collab = False

    for line in lyrics:

        if "Read More" in line :
            paroles = []
            continue

        if (("Contributor" in line) or (line.strip() == "") or ("Lyrics" in line)) :#We don't need this type of info
            continue              
                
        #Get the text and perform small corrections
        line = re.sub(r"\s+([',;:.!?])", r"\1", line) #If a space before ponctuation remove space
        line = re.sub(r"\([^)]*\)", "", line)       # retire les backs
        if line == "" :
            continue
            #Part where we resolve the anomalies in the format
 
            #This part is specifically design to handle long part indication where the part is divided in multiple lines
            #Sometimes a bracket [that indicates a part] is on multiple lines, when multiple artist
        if "[" in line and "]" not in line : 
            inside_bracket = True #Indicate that we are inside a bracket
            if "Paroles" in line : 
                parole_bracket = True
                continue
            paroles.append(line)
            continue
        elif "[" in line and "]" in line and "Paroles" in line :
            continue

        if inside_bracket  : 
            if parole_bracket : 
                if ']' in line :
                    parole_bracket = False
                    inside_bracket = False
                continue
            else :
                paroles[-1] += " "+line #If inside a bracket then everything is on the same line
                if "]" in line : #Close the bracket
                    inside_bracket = False
                continue #Skip the whole next part of if statement

        elif len(paroles)>0 and ((line[0] in [",", " "]) or (paroles[-1][-1] == "'")) : #If new line begin or previous line ended with , ==> not finished
            paroles[-1] += line 
            
        elif len(paroles)>0 and ((line[0].islower()) or (len(line)==1) or (paroles[-1][-1] == ",")) : #If the "new" line begin with a lowercase it indicate that the sentence wasn't finished
            paroles[-1] += " "+line
            
        else : 
            paroles.extend([line])

    del src, new
    gc.collect()
    return paroles, collab, session

def Navigate_songs(songs_list, artist_name):
    """
    Retrieve and structure lyrics for multiple Genius songs.

    Iterates through song links, extracts lyrics with `Get_lyrics_genius`, and
    formats them with section markers (e.g., <COUPLET>, <REFRAIN>). Skips
    non-lyrical or incomplete entries and reconstructs empty refrains when needed.

    Args:
        songs_list (list[bs4.element.Tag]): List of song link elements from Genius.
        artist_name (str): Target artist name for filtering collaborations.

    Returns:
        dict[str, list[str]]: Mapping of song titles to cleaned and structured lyrics.
    """

    dict_parole = {}
    session = requests.Session()
    skipped_songs = 0

    for songs_link in songs_list :
        true_link = songs_link[0]

        if "lyric" not in true_link : #or "translations" in true_link :
            skipped_songs += 1 
            continue

        lyrics, collab, session = Get_lyrics_genius(true_link, artist_name, session)

        if len(lyrics) < 4 :
            continue

        only_lyrics = []

        begin_music = True
        first_part = True
        SECTION_TOKENS = {
            "Couplet": "<COUPLET>",
            "Refrain": "<REFRAIN>",
            "Intro": "<INTRO>",
            "Outro": "<OUTRO>",
            "Pont": "<PONT>",
        }
        
        actual_part = "" #Will be used to close the part
        for i in lyrics: #A ce stade, les indications de couplet/refrains ont toujours là entre [] et les backs aussi entre ()
            i = i.strip()

            if begin_music :
                only_lyrics.append("<BEGINNING_SONG>"+str(collab))
                begin_music = False
            
            for key, token in SECTION_TOKENS.items() :
                if re.search(rf"\b{key}\b", i) : #If i use .lower there can be a problem where the word appear in the lyrics so a new part is created where there is none (happened with intro)
                    if not first_part:
                        only_lyrics.append(f"<END_{actual_part.upper()}>")

                    only_lyrics.append(f"{token}")
                    actual_part = key
                    first_part = False
                    continue
            
            if (i[0] != "[" and i[-1] != "]")  :
                only_lyrics.append(i)    

        if actual_part != "":
            only_lyrics.append(f"<END_{actual_part.upper()}>")
        only_lyrics.append("<END_SONG>\n")
        
        ## Part that resolve the pb where a part (REFRAIN) has no lyrics because it is identical has the previous one (in older genius's lyrics)
        refrain_part = []
        where_to_add = []
        
        for n in range(len(only_lyrics)-1) :
            if (only_lyrics[n] == "<REFRAIN>") and (only_lyrics[n+1] != "<END_REFRAIN>") : #Identify a complete REFRAIN
                j = n + 1
                while only_lyrics[j] != "<END_REFRAIN>" : #Collect all the lyrics of the REFRAIN
                    refrain_part.append(only_lyrics[j])
                    j += 1

            if (only_lyrics[n] == "<REFRAIN>") and (only_lyrics[n+1] == "<END_REFRAIN>") : #Collect the empty REFRAIN
                where_to_add.append(n+1)

        compteur_add = 0 #Complete the empty REFRAIN
        for i in where_to_add :
            for j in refrain_part : #If use of refrain_part directly, we have nested list 
                only_lyrics.insert(i+compteur_add,j)
                compteur_add += 1
                    
        song_title = songs_link[1]

        nb_parts = len(re.findall("<END.*", "\n".join(only_lyrics))) - 1 #The end is counted as a part
        length = len("\n".join(only_lyrics))

        #Will skipped songs with > 10 parts and also >= 1
        if nb_parts <= 10 and nb_parts >= 1 and length > 500 :
            dict_parole[f"{song_title}"] = only_lyrics
        else : 
            skipped_songs += 1

    print("Songs skipped due to non conform format or other problem :", skipped_songs)
    
    return dict_parole

def prepare_lyrics(artist_name, title_set):
    """
    Retrieve, clean, and compile lyrics into RNN-ready and tokenization corpora.

    Uses `Navigate_songs` to collect lyrics, cleans unwanted characters and tags,
    removes empty entries, and writes two text corpora: one preserving structure
    (for RNNs) and one normalized (for tokenization or embeddings).

    Args:
        artist_name (str): Artist name.
        title_set (set[str]): Song titles to process.

    Returns:
        None
    """

    result = Navigate_songs(title_set, artist_name)

    #Cleaning of lyrics 
    for title in list(result.keys()): 
        lines = result[title] #Je suis obligé d'itérer sur une copie et nan le vrai dict car je modifie sa valeur si len(lines) == 0
    
        lines = [line.replace("\u2005", " ") for line in lines]
        lines = [line.replace("[", "") for line in lines]
        lines = [line.replace("]", "") for line in lines]
    
        result[title] = lines
    
    #Regroupement de l'entiéreté du corpus de texte
    corpus = []

    for title in result :
        for elem in result[title] :
            corpus.extend([elem])

    #Préparation à la tokenization
    corpus_RNN = "\n".join(corpus)
    corpus_RNN = corpus_RNN.replace('"',"")

    corpus_tokenization = re.sub(r'[?,:/\.\(\)]', '', corpus_RNN)
    corpus_tokenization = re.sub(r'<[^>]*>', '', corpus_tokenization)
    corpus_tokenization = corpus_tokenization.lower()
    corpus_tokenization = re.sub(r'\n{2,}', '\n', corpus_tokenization)

    with open(f"Corpus/corpus_RNN_{artist_name}.txt", "w", encoding="utf-8") as f:
        f.write(corpus_RNN)
#    print(f"\nFind corpus usable for RNN at corpus_RNN_{artist_name}.txt")

#    with open(f"corpus_tokenization_{artist_name}.txt", "w", encoding="utf-8") as f:
#        f.write(corpus_tokenization)
#    print(f"Find corpus usable for tokenization at corpus_tokenization_{artist_name}.txt")

    return 

parser = argparse.ArgumentParser()
parser.add_argument("--char", type=bool, help="Perform specific pre-processing task", required=True)
parser.add_argument("--subword", type=bool, help="Perform specific pre-processing task", required=True)
parser.add_argument("--nb", type=int, help="Number of artists that will be collected", required=True)

args = parser.parse_args()
char = args.char
subword = args.subword

with open("API_genius.txt","r",encoding="utf-8") as f :
    GENIUS_API_TOKEN = f.read()
headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}


def process_all(artist_infos) :

    artist_name, artist_id = artist_infos
    songs = new_find_discography(artist_name, artist_id)
    prepare_lyrics(artist_name, songs)
    subprocess.run(["python", "Corpus/Cleaning_txt.py", "--name", artist_name])

if args.nb == 1 :
    result = get_the_artist_id(headers) 

    if result is None :  #If error during retrieving artist id, stop
        sys.exit()
        
    artist_name, artist_id = result        
    songs = new_find_discography(artist_name, artist_id)
    prepare_lyrics(artist_name, songs)
    subprocess.run(["python", "Corpus/Cleaning_txt.py", "--name", artist_name])
    subprocess.run(["python", "Markov/Markov_Chain.py"])
    
    if char == True :
        subprocess.run(["python", "Corpus/Manip_char.py", "--type", "char"])
    if subword == True :
        subprocess.run(["python", "Corpus/Manip_subword.py", "--type", "subword"])


elif args.nb > 1 :

    all_artists = []
    for link in range(args.nb) :

        result = get_the_artist_id(headers) 

        if result is None :  #If error during retrieving artist id, skip
            continue
        
        artist_name, artist_id = result    
        all_artists.append([artist_name, artist_id])
    
    if __name__ == '__main__':
        with multiprocessing.Pool(processes=3) as pool : #limit the nb of process to avoid getting rate limited
            for _ in tqdm(pool.imap_unordered(process_all, all_artists), total=len(all_artists)):
                pass

    subprocess.run(["python", "Markov/Markov_Chain.py"])

    if char == True :
        subprocess.run(["python", "Corpus/Manip_char.py", "--type", "char"])
    if subword == True :
        subprocess.run(["python", "Corpus/Manip_subword.py", "--type", "subword"])