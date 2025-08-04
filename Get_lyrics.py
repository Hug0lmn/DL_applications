from bs4 import BeautifulSoup as bs
import requests
import json
import re
from tqdm import tqdm
import argparse
import fasttext

#All selenium import
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib

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

    
    #Get the html page
    request = requests.get(url)
    html_page = request.text
    page = bs(html_page,"html.parser")

    #Get the artist name
    artist_name = page.find("h1", attrs={"id" : "naboo_artist_name"}).span.get_text(strip=True)
    if artist_name is None :
        raise ValueError("Artist name not found")
    else :
        print(f"Found the artist : {artist_name}")


    #Collect all the discography
    all_info_html = page.find("script", string = re.compile("ART_ID"))
    all_info = all_info_html.text.split("window.__DZR_APP_STATE__ = ")[1]
    structured_info = json.loads(all_info)

    structured_discography = structured_info["TOP"]["data"]
    if structured_discography is None :
        raise ValueError("Discography not found")
    else : 
        print("Found artist discography")

    #Organize all the data
    title_set = set() #Get all the title on what the artist performed
    album_set = set() #Get all the album that the artist is on
    collab_set = set() #Get all the possible collab -1
    time_count = 0

    for title in structured_discography :

        album_set.add(title["ALB_TITLE"])
        time_count += int(title["DURATION"])

        if len(title["ARTISTS"])>1 : 
            title_set.add(title["SNG_TITLE"])
            for artist in title["ARTISTS"] :
                collab_set.add(artist["ART_NAME"])
        else :
            title_set.add(title["SNG_TITLE"])
    
    return artist_name,title_set

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
def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def Get_lyrics_genius (driver) :

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


    src = driver.page_source
    new = bs(src,"html.parser")

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
                author_lyrics = "Limsa" in elem.text
            
            if collab and not author_lyrics : #Passe au prochain elem jusqu'à ce qu'un élément corresponde à author_lyrics
                continue

            if elem.name == "br":
                    # Si <br> est seul (ligne vide), on saute
                if ligne.strip():
                    paroles.append(ligne.strip())
                ligne = ""
            elif elem.string:
                ligne += elem.string
            elif elem.name == "i" and "class" not in str(elem): #Doit exclure des class qui sont contenus dans des names <i>
                #Certains textes en italique ont alors l'indication <i> example : 
                #<i>Qui a encore un peu d’espoir ?<br/>J’préfère être mal accompagné plutôt qu’seul au sommet dans le brouillard</i>
                #Dans la section en italique, les sauts de ligne ne sont pas automatique dans le texte mais indiqué par <br/>
                tnf = re.sub(r"</?i\s*/?>", "", str(elem)) #retire <i> / </i> / </i/> / <i/>
                dnc = tnf.split("<br/>")
                for i in dnc : 
                    paroles.append(i.strip())
    # Ajouter la dernière ligne si elle existe
        if ligne.strip():
            paroles.append(ligne.strip())
    
    return paroles

def Navigate_Web (artist_name, title_list) :

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
    found_site = 0
    no_found = []

    print("Begin the lyrics search...")
    
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")  # Optional
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    for idx, title in tqdm(enumerate(title_list), total=len(title_list)):
        
        target_song = artist_name + " " + str(title)
        target_site = "https://genius.com"
        query = f"{target_song} site:{target_site}"
        url = "https://duckduckgo.com/?q=" + urllib.parse.quote(query)
        
        driver.get(url) #Access the link
        
        WebDriverWait(driver, 5).until( #Wait until this specific element is located (for link)
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a'))
        )
        results = driver.find_elements(By.CSS_SELECTOR, 'a') #Store all the results

#        target = "https://"+target_site
        found = False

        #Preparation du titre 
        title_adapt = re.sub(r"\([^)]*\)" , "" , (re.sub(r"[\'\.\-\°]", "", title))).lower()        
#        title_adapt = re.sub(r"\([^)]*\)", "", title.lower())
        pattern = re.sub(r"\s+", r"[-_+%20]*", (remove_accents(title_adapt)))

        for link in results :
            href = link.get_attribute("href")
            if href :
                last_word_link = href.split("-")[-1]
                
            if href and (target_site in href) and ((re.search(pattern,href.lower())) or (re.search(title.lower(),href.lower()))) and last_word_link == "lyrics" :
                link.click()
                found = True
                found_site +=1
                break
        
        if not found : #Site Genius musique non trouvé
            no_found.append(title)
            found = False
            continue
    
        if idx == 0 : 
            Accept_cookies_genius(driver)

        lyrics = Get_lyrics_genius(driver)

        only_lyrics = []
        for i in lyrics: #A ce stade, les indications de couplet/refrains ont toujours là entre [] et les backs aussi entre ()
            i = re.sub(r"\([^)]*\)", "", i)       # retire les backs
            indication = ("couplet" in i.lower()) or ("refrain" in i.lower()) or ("intro" in i.lower()) or ("paroles" in i.lower()) 
            if i and indication == False: #Contains at least an element
                only_lyrics.append(i)    
        
        dict_parole[f"{title}"] = only_lyrics
    
    print("End of lyrics search")
    print(f"{found_site} on {len(title_list)} songs found")
    for i in no_found : 
        print(f"Lyrics not found for {i}")
    driver.quit()

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

    result = Navigate_Web(artist_name, title_set)

    #Cleaning of lyrics 
    for title in list(result.keys()): 
        lines = result[title] #Je suis obligé d'itérer sur une copie et nan le vrai dict car je modifie sa valeur si len(lines) == 0
    
        lines = [line.replace("\u2005", " ") for line in lines]
        lines = [line.replace("[", "") for line in lines]
        lines = [line.replace("]", "") for line in lines]
    
        result[title] = lines

        if not lines:
            del result[title]
            print(f"Deleted for missing lyrics : {title}")
    
    #Regroupement de l'entiéreté du corpus de texte
    corpus = []

    for title in result :
        for elem in result[title] :
            corpus.extend([elem])

    #Préparation à la tokenization
    corpus_RNN = "\n".join(corpus)
    corpus_RNN = corpus_RNN.replace('"',"")
    corpus_tokenization = re.sub(r'[?,:/\.]', '', corpus_RNN)

    with open("corpus_RNN.txt", "w", encoding="utf-8") as f:
        f.write(corpus_RNN)
    print("Find corpus usable for RNN at corpus_RNN.txt")

    with open("corpus_tokenization.txt", "w", encoding="utf-8") as f:
        f.write(corpus_tokenization)
    print("Find corpus usable for tokenization at corpus_tokenization.txt")

    return 

parser = argparse.ArgumentParser()
parser.add_argument("--link", type=str, help="Link of deezer's artist top songs", required=True)

args = parser.parse_args()
link = args.link

artist_name, title_set = Find_artist_discography(link)
prepare_lyrics(artist_name, title_set)
