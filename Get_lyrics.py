from bs4 import BeautifulSoup as bs
import requests
import json
import re
from tqdm import tqdm
import time
import argparse
import gc

#All selenium import
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import unicodedata

def Find_artist_discography(url) :

    """
    Extracts the artist name and full discography from a given Deezer top track URL.

    Given a URL to the artist’s top track on Deezer, this function retrieves the artist's name 
    and compiles a set of all song titles associated with the artist.

    Args:
        url (str): The Deezer URL pointing to the artist’s top track 
               (e.g., "https://www.deezer.com/fr/artist/14659541/top_track").

    Returns:
        Tuple[str, set]: A tuple containing:
            - artist_name (str): The name of the artist.
            - title_set (set): A set of all unique song titles found for the artist.

    Raises:
        ValueError: If the URL is malformed or if artist data could not be extracted.
        requests.exceptions.RequestException: If the HTTP request fails.

    Example:
        artist_name, songs = extract_artist_data("https://www.deezer.com/fr/artist/14659541/top_track")
    """

    #Navigate to html page
    options = webdriver.ChromeOptions()
    #options.add_argument("--headless=new")
    options.add_argument("--start-maximized")  
    
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url+"/songs")

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
            time.sleep(0.5)
        
        page = bs(driver.page_source,"html.parser")
        new_len = len(page.find("ul", {"class":"ListSection-desktop__Items-sc-2bca79e6-8 kVtuqy"}).find_all("li"))
        
        if new_len == songs_len :
            scroll = False
            songs = page.find("ul", {"class":"ListSection-desktop__Items-sc-2bca79e6-8 kVtuqy"}).find_all("li")
            print(f"Nb of songs found {new_len}")
        else : 
            songs_len = new_len

        nb_scroll = 0

    driver.quit()

    return artist_name, songs

def Accept_cookies_genius (driver) :

    """
    Handles the cookie consent popup on Genius using Selenium.
    It doesn't handle other website but can be tweak easily.

    This function waits for and clicks two cookie-related buttons on a webpage:
    1. "Afficher toutes les finalités" (Show all purposes)
    2. "Confirmer la sélection" (Confirm selection)

    If any of the buttons are not found or clickable within the timeout period, the function quits the Selenium driver and prints a message.

    Parameters:
    driver (selenium.webdriver): An active Selenium WebDriver instance controlling a browser.

    Returns:
    None
    """

    try:
        wait = WebDriverWait(driver, 10)
        button_cookies = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[text()="Afficher toutes les finalités"]')))
        button_cookies.click()
    except Exception as e :
        driver.quit()
        print("No button found")

    try:
        wait = WebDriverWait(driver, 10)
        button_cookies = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[text()="Confirmer la sélection"]')))
        button_cookies.click()
    except Exception as e :
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

def Get_lyrics_genius (link, artist_name) :

    """
    Extracts lyrics from a Genius song page using Selenium and BeautifulSoup.

    This function retrieves the HTML source from the current page loaded in the Selenium driver, parses it with BeautifulSoup,
    and extracts all text lines from the lyrics blocks. It handles special formatting cases such as `<br>` tags, `<i>` italics, 
    and collaborative tracks (e.g., where multiple artists are involved with labeled verses).

    Parameters:
    driver (selenium.webdriver): An active Selenium WebDriver instance currently on a Genius lyrics page.

    Returns:
    list of str: A list of cleaned lines of lyrics extracted from the page.
    """


    src = requests.get(link).text
    new = bs(src,"html.parser")
    artist_name = remove_accents_and_quotes(artist_name)

    # On récupère tous les blocs qui contiennent les paroles
    lyrics_blocks = new.find_all("div", attrs={"data-lyrics-container": "true"})
    paroles = []
    author_lyrics = False
    collab = False

    if new.find_all("span", string = "Featuring") :
        collab = True
    else : 
        collab = False

    for block in lyrics_blocks:
        ligne = ""
        for elem in block.children:

            if collab and "[" in elem.text and ":" in elem.text:
                author_lyrics = (artist_name.split(" ")[0] in remove_accents_and_quotes(elem.text)) and "&" not in elem.text
            
            if collab and not author_lyrics : #Passe au prochain elem jusqu'à ce qu'un élément corresponde à author_lyrics
                continue

            lyric = elem.get_text(separator="\n")
            if lyric != '' :
                paroles.extend(lyric.split("\n"))
    
    # Ajouter la dernière ligne si elle existe
        if ligne.strip():
            paroles.append(ligne.strip())

    del src, new
    gc.collect()
    
    return paroles

def Navigate_songs (songs_list, artist_name) :

    """
    Scrapes lyrics from Genius using DuckDuckGo search and Selenium.

    This function takes an artist name (tested on rapper) and a list of song titles, performs web searches for each track on Genius.com via DuckDuckGo,
    navigates to the corresponding Genius page if found, and extracts the lyrics. It handles cookie confirmation, lyric formatting cleanup, 
    and tracks which songs were not found.

    Parameters:
    -----------
    artist_name : str
        The name of the artist whose songs are being searched.
    title_list : list of str
        A list of song titles to retrieve lyrics for.

    Returns:
    --------
    dict_parole : dict
        A dictionary mapping each song title to a list of cleaned lyrics lines (list of str).
        Songs not found are excluded from the dictionary.

    Notes:
    ------
    - Uses Selenium WebDriver with Chrome, controlled via `webdriver-manager`.
    - Skips songs for which no Genius link is found or if lyrics cannot be extracted.
    - Handles cookie pop-ups on the Genius site.
    - Cleans lyrics by removing annotations such as "(backs)" or "[refrain]".
    """

    dict_parole = {}

    print("Begin the lyrics scrapping...")
    for songs_link in songs_list :
        true_link = songs_link.a.attrs["href"]
#        driver.get(true_link)
        lyrics = Get_lyrics_genius(true_link, artist_name)

        only_lyrics = []
        for i in lyrics: #A ce stade, les indications de couplet/refrains ont toujours là entre [] et les backs aussi entre ()
            i = re.sub(r"\([^)]*\)", "", i)       # retire les backs
            indication = ("couplet" in i.lower()) or ("refrain" in i.lower()) or ("intro" in i.lower()) or ("paroles" in i.lower()) 
            if i and indication == False: #Contains at least an element
                only_lyrics.append(i)    
        
        song_title = songs_link.a.h3.text
        dict_parole[f"{song_title}"] = only_lyrics
    
    print("End of lyrics search")
#    driver.quit()

    return dict_parole

def prepare_lyrics(artist_name, title_set) :

    """
    Retrieves, cleans, and prepares lyrics for both RNN training and tokenization-based models.

    This function scrapes song lyrics from Genius.com for a given artist and list of titles,
    cleans the text (removing unwanted characters and formatting), and writes two versions 
    of the corpus to text files:
    - A version with newlines and quotes preserved (for RNN training).
    - A version with additional punctuation removed (for word-level tokenization).

    Parameters:
    -----------
    artist_name : str
        The name of the artist whose lyrics are to be retrieved.
    
    title_set : set of str
        A set of song titles by the artist.

    Effects:
    --------
    - Saves `corpus_RNN.txt` : cleaned corpus preserving line breaks (for RNNs).
    - Saves `corpus_tokenization.txt` : cleaned and stripped version (for tokenization).
    - Prints status messages and any removed songs due to missing lyrics.

    Returns:
    --------
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
    
        result[title] = lines

        if not lines:
            del result[title]
            found_site += 1
            no_found.append(title)
            print(f"Deleted for missing lyrics : {title}")
    print(found_site, no_found)

    #Regroupement de l'entiéreté du corpus de texte
    corpus = []

    for title in result :
        for elem in result[title] :
            corpus.extend([elem])

    #Préparation à la tokenization
    corpus_RNN = "\n".join(corpus)
    corpus_RNN = corpus_RNN.replace('"',"")
    corpus_tokenization = re.sub(r'[?,:/\.]', '', corpus_RNN)

    with open(f"corpus_RNN_{artist_name}.txt", "w", encoding="utf-8") as f:
        f.write(corpus_RNN)
    print(f"Find corpus usable for RNN at corpus_RNN_{artist_name}.txt")

    with open(f"corpus_tokenization_{artist_name}.txt", "w", encoding="utf-8") as f:
        f.write(corpus_tokenization)
    print(f"Find corpus usable for tokenization at corpus_tokenization_{artist_name}.txt")

    return 

parser = argparse.ArgumentParser()
parser.add_argument("--link", type=str, help="Link of genius's artist page", nargs="+", required=True)

args = parser.parse_args()
links = args.link
for link in links : 
    artist_name, songs = Find_artist_discography(link)
#songs_lyrics = Navigate_songs(songs, driver)
#Next things to try will be multi threading to reduce heavily runtime

    prepare_lyrics(artist_name, songs)
