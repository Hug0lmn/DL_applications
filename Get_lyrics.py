from bs4 import BeautifulSoup as bs
import requests
import undetected_chromedriver as uc
import re
from bs4.element import Tag
from tqdm import tqdm
import time
import argparse
import gc
import subprocess

#All selenium import
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import unicodedata

def Find_artist_discography(driver, url):
    """
    Scrapes an artist's full discography from a Genius page.

    This function uses Selenium to navigate through the artist's Genius page,
    automatically accepts cookies, scrolls to dynamically load all songs, 
    and extracts their complete list of tracks.

    Args:
        url (str): Genius URL pointing to the artist’s page.
                   Example: "https://genius.com/artists/Limsa-daulnay"

    Returns:
        tuple[str, list]: 
            - artist_name (str): The name of the artist as shown on Genius.
            - songs (list): A list of BeautifulSoup tag elements representing
              each song entry.

    Raises:
        ValueError: If the artist name cannot be found on the page.
        requests.exceptions.RequestException: If there are issues retrieving the page.
        selenium.common.exceptions.WebDriverException: If Selenium fails to launch or navigate.

    Notes:
        - Requires Chrome, ChromeDriver, and the `webdriver_manager` package.
        - Runs Chrome in headless mode for faster scraping.
        - Scrolling is necessary because Genius loads songs lazily; 
          this function scrolls until no new songs are loaded.
    """

    #Navigate to html page
#    options = webdriver.ChromeOptions()
#    options.add_argument("--headless=new")
#    options.add_argument("--start-maximized")  
    
    no_cookies = False 

    if driver == False :
        options = uc.ChromeOptions()
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        driver = uc.Chrome(options=options)
    else : 
        no_cookies = True


    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url+"/songs")

    WebDriverWait(driver, 20).until(
    lambda d: d.execute_script("return document.readyState") == "complete"
    )

    if no_cookies == False :
        Accept_cookies_genius(driver)

        #Get the html page
    page = bs(driver.page_source,"html.parser")

    #Get the artist name
    artist_name = page.find("a", {"href" : url}).text
    if artist_name is None :
        raise ValueError("Artist name not found")
    else :
        print(f"Found the artist : {artist_name}")

    #Scroll down to have all the songs displayed (all songs aren't displayed automatically, you need to scroll down to force the load)
    songs = None #Will have the entire songs as a list
    songs_len = len(page.find("ul", {"class":"ListSection-desktop__Items-sc-2bca79e6-8 kVtuqy"}).find_all("li"))
    scroll = True

    while scroll == True :
        nb_scroll = 0
        
        while nb_scroll < 5 :
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            nb_scroll += 1
            time.sleep(1)
        
        page = bs(driver.page_source,"html.parser")
        new_len = len(page.find("ul", {"class":"ListSection-desktop__Items-sc-2bca79e6-8 kVtuqy"}).find_all("li"))
        
        if new_len == songs_len :
            scroll = False
            songs = page.find("ul", {"class":"ListSection-desktop__Items-sc-2bca79e6-8 kVtuqy"}).find_all("li")
            print(f"Nb of songs found {new_len}")
        else : 
            songs_len = new_len

        nb_scroll = 0

#    driver.quit()

    return artist_name, songs, driver

def Accept_cookies_genius(driver):
    """
    Handles the Genius.com cookie consent popup using Selenium.

    This function automates the acceptance of cookies by clicking on two buttons:
    1. "Afficher toutes les finalités" (Show all purposes)
    2. "Confirmer la sélection" (Confirm selection)

    It waits up to 10 seconds for each button to become clickable.
    If a button cannot be found or clicked within the timeout, the Selenium driver 
    is closed and a message is printed. This function is tailored for Genius.com, 
    but can be adapted for other websites by updating the XPaths.

    Args:
        driver (selenium.webdriver): An active Selenium WebDriver instance controlling a browser.

    Returns:
        None

    Raises:
        selenium.common.exceptions.TimeoutException: If either button is not found within the timeout.
        selenium.common.exceptions.WebDriverException: If there is an issue interacting with the browser.

    Notes:
        - This function is specific to Genius.com's cookie popup in French.
    """

    for attempt in range(2):  
        try:
            wait = WebDriverWait(driver, 10)
            button_cookies = wait.until(
                EC.element_to_be_clickable((By.XPATH, '//button[text()="Afficher toutes les finalités"]'))
            )
            button_cookies.click()
            break  
        except Exception as e:
            if attempt != 0:  
                driver.quit()
                print("No button found")

    for attempt in range(2):  
        try:
            wait = WebDriverWait(driver, 10)
            button_cookies = wait.until(
                EC.element_to_be_clickable((By.XPATH, '//button[text()="Confirmer la sélection"]'))
            )
            button_cookies.click()
            break
        except Exception as e:
            if attempt != 0:
                driver.quit()
                print("No button found")

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

def Get_lyrics_genius(link, artist_name):
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
        link (str): The Genius URL of the song's lyrics page.
        artist_name (str): The main artist's name; used to filter 
            verses in collaborative tracks so only their parts are returned.

    Returns:
        list[str]: A list of cleaned lyric lines for the specified artist.
                   If the song is a collaboration, only the main artist's
                   sections are returned (when detectable).

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
        ValueError: If no lyrics container is found in the page HTML.

    Notes:
        - Collaborator detection is based on the presence of "Featuring" 
          in the page and section headers that contain the artist name.
        - This function does not require Selenium; it fetches the HTML directly.
        - Punctuation spacing is normalized: e.g., `" , "` → `","` and `" . "` → `". "`.

    """

    src = requests.get(link).text
    new = bs(src,"html.parser")
    artist_name = remove_accents_and_quotes(artist_name)

    # On récupère tous les blocs qui contiennent les paroles
    lyrics_blocks = new.find_all("div", attrs={"data-lyrics-container": "true"})
    paroles = []
    collab = False #Indication if the music is a featuring, if so the code will need to check if each part was written by our author
    author_lyrics = False #Indication if the lyrics were written by our specified artist, only if collab = True
#    annot = False #Indication if the line of lyrics contains a commentary, used because this can separate a sentence into multiple lines
    inside_bracket = False 

    if new.find_all("span", string = "Featuring") : #Identify a collab (used for post processing and specific scrapping treatment)
        collab = True
    else : 
        collab = False

    for block in lyrics_blocks:
        ligne = ""
        for elem in block.children: #Most of the time the lyrics are divided in separate parts

            txt = elem.get_text()
            if (("Contributor" in txt) or ("Paroles" in txt)) or (txt.strip() == "") :#We don't need this type of info
                continue              
                
            list_part = ["Couplet","Refrain","Intro","Outro","Pont"]
            #Featuring part
            if collab and (('[' in txt[0] or '(' in txt[0])) and any(re.search(rf"{part.lower()}", txt.lower()) for part in list_part): #If the text indicates a part in a featuring
                author_lyrics = (artist_name.split(" ")[0] in remove_accents_and_quotes(txt)) and "&" not in txt #If the artist name is found in this text and only one name : author_lyrics == True
                paroles.append(txt)
                continue

            if collab and not author_lyrics : #If the part isn't sung by our artist, skip the text
                continue

            #Get the text and perform small corrections
            text = elem.get_text(separator="<br>").strip() #The separator will be used to split the text, sometimes two lines are regroup into one but can be splitted later
            text = re.sub(r"\s+([',;:.!?])", r"\1", text) #If a space before ponctuation remove space
            text = re.sub(r"\([^)]*\)", "", text)       # retire les backs
            if text == "" :
                continue
            #Part where we resolve the anomalies in the format
 
            #This part is specifically design to handle long part indication where the part is divided in multiple lines
            #Sometimes a bracket [that indicates a part] is on multiple lines, when multiple artist
            if "[" in text and "]" not in text : 
                inside_bracket = True #Indicate that we are inside a bracket
                paroles.append(text)
                continue

            if inside_bracket  : 
                paroles[-1] += " "+text #If inside a bracket then everything is on the same line
                if "]" in text : #Close the bracket
                    inside_bracket = False
                continue #Skip the whole next part of if statement

            if "<br>" in text : 
                for line in text.split("<br>") :
                    line = line.strip()

                    if len(line) == 0 :
                        continue
                
                    if len(paroles) == 0 : #This part is of use when there is only one lyrics_blocks
                        paroles.extend([line])
                        continue

                    #This part is of use in a normal lyrics_blocks when inside an annotation, that concat multiple lines
                    if (line[0].islower()) or (len(line)==1) : #If the first letter is a lowercase or contains only one char
                        paroles[-1] += " "+line #we assume the sentence wasn't finished, we add it to the previous line 

                    elif line[0] == "," : #If the new line begin by ,
                        paroles[-1] += line

                    else : #Else 
                        if paroles[-1][-1] in [",","&"] : #This is use because on a number of outro, i observed that the outro part is divided in multiple lines when there is multiple artists in it
                            paroles[-1] += " "+line
                        else :
                            paroles.extend([line])

            elif len(paroles)>0 and ((text[0] in [",", " "]) or (paroles[-1][-1] == "'")) : #If new line begin or previous line ended with , ==> not finished
                paroles[-1] += text 
            
            elif len(paroles)>0 and ((text[0].islower()) or (len(text)==1) or (paroles[-1][-1] == ",")) : #If the "new" line begin with a lowercase it indicate that the sentence wasn't finished
                paroles[-1] += " "+text
            
            else : 
                paroles.extend([text])

    del src, new
    gc.collect()
    return paroles, collab

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

    #print("Begin the lyrics scrapping...")
    for songs_link in tqdm(songs_list, desc='Scrapping songs'):
        true_link = songs_link.a.attrs["href"]
#        driver.get(true_link)
        if "lyric" not in true_link : 
            continue

        lyrics, collab = Get_lyrics_genius(true_link, artist_name)

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
        
        for i in lyrics: #A ce stade, les indications de couplet/refrains ont toujours là entre [] et les backs aussi entre ()
            i = i.strip()

            if begin_music :
                only_lyrics.append("<BEGINNING>"+str(collab))
                begin_music = False
            
            for key, token in SECTION_TOKENS.items() :
                if re.search(rf"\b{key}\b", i) : #If i use .lower there can be a problem where the word appear in the lyrics so i new part is created where there is none (happened with intro)
                    if not first_part:
                        only_lyrics.append("<END_SECTION>")
                    only_lyrics.append(f"{token}")
                    first_part = False
                    continue
            
            if (i[0] != "[" and i[-1] != "]")  :
                only_lyrics.append(i)    

        only_lyrics.append("<END_SECTION>")
        only_lyrics.append("<END>\n")
        
        ## Part that resolve the pb where a part (REFRAIN) has no lyrics because it is identical has the previous one 
        refrain_part = []
        where_to_add = []
        
        for n in range(len(only_lyrics)-1) :
            if (only_lyrics[n] == "<REFRAIN>") and (only_lyrics[n+1] != "<END_SECTION>") : #Identify a complete REFRAIN
                j = n + 1
                while only_lyrics[j] != "<END_SECTION>" : #Collect all the lyrics of the REFRAIN
                    refrain_part.append(only_lyrics[j])
                    j += 1

            if (only_lyrics[n] == "<REFRAIN>") and (only_lyrics[n+1] == "<END_SECTION>") : #Collect the empty REFRAIN
                where_to_add.append(n+1)

        compteur_add = 0 #Complete the empty REFRAIN
        for i in where_to_add :
            for j in refrain_part : #If use of refrain_part directly, we have nested list 
                only_lyrics.insert(i+compteur_add,j)
                compteur_add += 1
                    
        song_title = songs_link.a.h3.text
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

    found_site = 0
    no_found = []

    #Cleaning of lyrics 
    for title in list(result.keys()): 
        lines = result[title] #Je suis obligé d'itérer sur une copie et nan le vrai dict car je modifie sa valeur si len(lines) == 0
    
        lines = [line.replace("\u2005", " ") for line in lines]
        lines = [line.replace("[", "") for line in lines]
        lines = [line.replace("]", "") for line in lines]
    
        if not lines:
            del result[title]
            found_site += 1
            no_found.append(title)
            print(f"Deleted for missing lyrics : {title}")

        result[title] = lines
    
    if found_site > 0 :
        print(found_site, no_found)

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
    print(f"Find corpus usable for RNN at corpus_RNN_{artist_name}.txt")

#    with open(f"corpus_tokenization_{artist_name}.txt", "w", encoding="utf-8") as f:
#        f.write(corpus_tokenization)
#    print(f"Find corpus usable for tokenization at corpus_tokenization_{artist_name}.txt")

    return 

parser = argparse.ArgumentParser()
parser.add_argument("--link", type=str, help="Link of genius's artist page", nargs="+", required=True)
parser.add_argument("--RNN", type=bool, help="Perform specific pre-processing task", required=True)

args = parser.parse_args()
links = args.link
rnn = args.RNN

print(args)

if len (links) == 1 : 
    artist_name, songs = Find_artist_discography(links[0])
    prepare_lyrics(artist_name, songs)
    subprocess.run(["python", "Corpus/Cleaning_txt.py", "--name", artist_name])
    subprocess.run(["python", "Markov/Markov_Chain.py"])

elif len(links) > 1 :
    artists_names = []
    songs_ = []
    driver = False

    for link in tqdm(links, desc='Scrapping Artist'):
        #Idea : try multi threading to reduce runtime  
        artist_name, songs, driver = Find_artist_discography(driver, link)
        artists_names.append(artist_name)
        songs_.append(songs)

    driver.quit()

    for i in range(len(artists_names)) :
        prepare_lyrics(artists_names[i], songs_[i])
        subprocess.run(["python", "Corpus/Cleaning_txt.py", "--name", artists_names[i]])
    subprocess.run(["python", "Markov/Markov_Chain.py"])

    if rnn == True :
        subprocess.run(["python", "Corpus/Manip.py"])
