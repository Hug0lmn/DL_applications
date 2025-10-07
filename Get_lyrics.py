from bs4 import BeautifulSoup as bs
import requests
import re
from tqdm import tqdm
import argparse
import gc
import subprocess
import unicodedata

def search_artist(query, headers): #Will send back 10 results 
    url = f"https://api.genius.com/search"
    params = {"q": query}
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def collect_all_songs(artist_id, page_nb, headers): #Will send back 20 songs of the artist_id
    url = f"https://api.genius.com/artists/{artist_id}/songs?per_page=20&page={page_nb}"
    response = requests.get(url, headers=headers)
    return response.json()

def new_find_discography(headers) :

    result_search = search_artist(input("\nName a song of your artist or the artist's name : "), headers)

    #Will trr to find the artist specific id in order to find their songs
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
        raise ValueError("Provide a valid number or change the song's name")

    ## Collect the artist's discography 
    artist_name = collect_artist[result][0]
    collect_songs = []
    
    for i in range(1,50) :
        songs_ = collect_all_songs(collect_artist[result][1], i, headers)

        if len(songs_["response"]["songs"]) == 0 : #No songs returned
            break
        collect_songs.extend(songs_["response"]["songs"])

    #Check if lyrics are good
    list_songs =[]
    for i in collect_songs : 
        if i["lyrics_state"] == "complete" :
            list_songs.append([i["url"],i["title"]])

    return list_songs, artist_name

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

def Get_lyrics_genius(link, artist_name, s):
    """
    Extracts and cleans lyrics from a Genius song page.

    This function downloads a Genius song page via HTTP (using `requests`), 
    parses the HTML with BeautifulSoup, and extracts all text.  
    It handles:
      - Removal of extra spaces and spacing before punctuation.
      - Collaborative tracks (e.g., "Featuring" credit) by filtering verses
        to include only those attributed to the target artist.
      - Line breaks and multiple spaces normalization.

    Args:

    Returns:

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
        ValueError: If no lyrics container is found in the page HTML.

    Notes:
        - Collaborator detection is based on the presence of "Featuring" 
          in the page and section headers that contain the artist name.
        - This function does not require Selenium; it fetches the HTML directly.
        - Punctuation spacing is normalized: e.g., `" , "` → `","` and `" . "` → `". "`.

    """

    src = s.get(link).text
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
    #lyrics_blocks = new.find_all("div", attrs={"data-lyrics-container": "true"})
    paroles = []
    
    collab = False #Indication if the music is a featuring, if so the code will need to check if each part was written by our author
#    author_lyrics = False #Indication if the lyrics were written by our specified artist, only if collab = True
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
                
#        list_part = ["Couplet","Refrain","Intro","Outro","Pont"]
        #Featuring part
#        if collab and (('[' in line[0] or '(' in line[0])) and any(re.search(rf"{part.lower()}", line.lower()) for part in list_part): #If the text indicates a part in a featuring
#            author_lyrics = (artist_name.split(" ")[0] in remove_accents_and_quotes(line)) and "&" not in line #If the artist name is found in this text and only one name : author_lyrics == True
#            paroles.append(line)
#            continue

#        if collab and not author_lyrics : #If the part isn't sung by our artist, skip the text
#            continue

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
    return paroles, collab, s

def Navigate_songs(songs_list, artist_name):
    """
    Retrieves and cleans lyrics for multiple songs from Genius.

    This function iterates through a list of song HTML elements, extracts the Genius URL for each track, 
    retrieves its lyrics via `Get_lyrics_genius`, and performs cleanup such as 
    removing annotations and marking structural elements (e.g., <COUPLET>, <REFRAIN>).

    Args:
        songs_list (list[bs4.element.Tag]): 
            A list of BeautifulSoup tag elements, each containing a song entry 
            with a link (`<a>` tag) to its Genius page.
        artist_name (str): 
            Name of the artist, used for filtering lyrics in collaborative tracks.

    Returns:
        dict[str, list[str]]: 
            Dictionary mapping song titles to a list of cleaned lyric lines.  
            Structural markers are added in uppercase between `<STROPHE>` tags:
              - `<COUPLET>` for verses
              - `<REFRAIN>` for choruses
              - `<INTRO>` for introductions
              - `<OUTRO>` for conclusions

    Notes:
        - Songs for which lyrics cannot be found are skipped.
        - Removes "backs" or background vocals inside parentheses `(...)`.
        - Ignores non-lyrical annotations like "(paroles)", "(contributors)", "(pont)", etc.
        - Structural markers are inserted as separate elements in the lyrics list.
    """


    dict_parole = {}
    s = requests.Session()

    #print("Begin the lyrics scrapping...")
    for songs_link in tqdm(songs_list, desc='Scrapping songs'):
        true_link = songs_link[0]
#        driver.get(true_link)
        if "lyric" not in true_link : 
            continue

        lyrics, collab,s = Get_lyrics_genius(true_link, artist_name, s)

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
        dict_parole[f"{song_title}"] = only_lyrics
    
    #print("End of lyrics search")
#    driver.quit()

    return dict_parole

def prepare_lyrics(artist_name, title_set):
    """
    Retrieves, cleans, and prepares lyrics for both RNN training 
    and word-level tokenization models.

    This function:
      1. Uses `Navigate_songs` to retrieve cleaned lyrics for each song title.
      2. Removes unwanted characters such as invisible spaces, brackets, and quotes.
      3. Deletes entries with no available lyrics and logs them.
      4. Aggregates all lyrics into a single corpus.
      5. Generates two versions of the corpus:
         - RNN corpus: preserves case and line breaks (only removes double quotes).
         - Tokenization corpus: lowercased, punctuation stripped, tags removed,
           and normalized line breaks.

    Args:
        artist_name (str): 
            Name of the artist whose lyrics are to be scraped and processed.
        title_set (set[str]): 
            Set of song titles by the artist.

    Effects:
        - Writes `corpus_RNN_<artist_name>.txt` for character/sequence models (RNN, LSTM, etc.).
        - Writes `corpus_tokenization_<artist_name>.txt` for embedding training (FastText, Word2Vec, etc.).
        - Prints the count and names of songs removed due to missing lyrics.

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
    print(f"\nFind corpus usable for RNN at corpus_RNN_{artist_name}.txt")

#    with open(f"corpus_tokenization_{artist_name}.txt", "w", encoding="utf-8") as f:
#        f.write(corpus_tokenization)
#    print(f"Find corpus usable for tokenization at corpus_tokenization_{artist_name}.txt")

    return 

parser = argparse.ArgumentParser()
#parser.add_argument("--link", type=str, help="Link of genius's artist page", nargs="+", required=True)
parser.add_argument("--RNN", type=bool, help="Perform specific pre-processing task", required=True)
parser.add_argument("--nb", type=int, help="Number of artists that will be collected", required=True)

args = parser.parse_args()
#links = args.link
rnn = args.RNN

with open("API_genius.txt","r",encoding="utf-8") as f :
    GENIUS_API_TOKEN = f.read()
headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}

if args.nb == 1 :
    songs, artist_name = new_find_discography(headers)
    prepare_lyrics(artist_name, songs)
    subprocess.run(["python", "Corpus/Cleaning_txt.py", "--name", artist_name])
    subprocess.run(["python", "Markov/Markov_Chain.py"])
    
    if rnn == True :
        subprocess.run(["python", "Corpus/Manip.py"])

elif args.nb > 1 :
    artists_names = []
    songs_ = []

    for link in range(args.nb) :
        #Idea : try multi threading to reduce runtime  
        songs, artist_name = new_find_discography(headers)
        artists_names.append(artist_name)
        songs_.append(songs)

#    driver.quit()
    for i in range(len(artists_names)):
        prepare_lyrics(artists_names[i], songs_[i])
        subprocess.run(["python", "Corpus/Cleaning_txt.py", "--name", artists_names[i]])
    subprocess.run(["python", "Markov/Markov_Chain.py"])

    if rnn == True :
        subprocess.run(["python", "Corpus/Manip.py"])
