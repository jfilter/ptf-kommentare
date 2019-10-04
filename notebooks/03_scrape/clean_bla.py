#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import Parallel, delayed
from math import sqrt
from pathlib import Path
from lxml import etree
from tqdm import tqdm
import pickle
import dateparser

from bs4 import BeautifulSoup

import pandas as pd
import swifter

import pandas as pd
import sqlite3
from cleantext import clean

from pathlib import Path
import numpy as np
import swifter
from somajo import Tokenizer, SentenceSplitter
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

from german_lemmatizer import lemmatize

num_cores = multiprocessing.cpu_count()

import numpy as np


# In[ ]:




sents = pickle.load( open( "s.pkl", "rb" ) )


# In[3]:






STOP_WORDS = set(
    """
á a ab aber ach acht achte achten achter achtes ag alle allein allem allen
aller allerdings alles allgemeinen als also am an andere anderen anderem andern
anders auch auf aus ausser außer ausserdem außerdem
bald bei beide beiden beim beispiel bekannt bereits besonders besser besten bin
bis bisher bist
da dabei dadurch dafür dagegen daher dahin dahinter damals damit danach daneben
dank dann daran darauf daraus darf darfst darin darüber darum darunter das
dasein daselbst dass daß dasselbe davon davor dazu dazwischen dein deine deinem
deiner dem dementsprechend demgegenüber demgemäss demgemäß demselben demzufolge
den denen denn denselben der deren derjenige derjenigen dermassen dermaßen
derselbe derselben des deshalb desselben dessen deswegen dich die diejenige
diejenigen dies diese dieselbe dieselben diesem diesen dieser dieses dir doch
dort drei drin dritte dritten dritter drittes du durch durchaus dürfen dürft
durfte durften
eben ebenso ehrlich eigen eigene eigenen eigener eigenes ein einander eine
einem einen einer eines einigeeinigen einiger einiges einmal einmaleins elf en
ende endlich entweder er erst erste ersten erster erstes es etwa etwas euch
früher fünf fünfte fünften fünfter fünftes für
gab ganz ganze ganzen ganzer ganzes gar gedurft gegen gegenüber gehabt gehen
geht gekannt gekonnt gemacht gemocht gemusst genug gerade gern gesagt geschweige
gewesen gewollt geworden gibt ging gleich gott gross groß grosse große grossen
großen grosser großer grosses großes gut gute guter gutes
habe haben habt hast hat hatte hätte hatten hätten heisst heißt her heute hier
hin hinter hoch
ich ihm ihn ihnen ihr ihre ihrem ihren ihrer ihres im immer in indem
infolgedessen ins irgend ist
ja jahr jahre jahren je jede jedem jeden jeder jedermann jedermanns jedoch
jemand jemandem jemanden jene jenem jenen jener jenes jetzt
kam kann kannst kaum kein keine keinem keinen keiner kleine kleinen kleiner
kleines kommen kommt können könnt konnte könnte konnten kurz
lang lange leicht leider lieber los
machen macht machte mag magst man manche manchem manchen mancher manches mehr
mein meine meinem meinen meiner meines mensch menschen mich mir mit mittel
mochte möchte mochten mögen möglich mögt morgen muss muß müssen musst müsst
musste mussten
na nach nachdem nahm natürlich neben nein neue neuen neun neunte neunten neunter
neuntes nicht nichts nie niemand niemandem niemanden noch nun nur
ob oben oder offen oft ohne
recht rechte rechten rechter rechtes richtig rund
sagt sagte sah satt schlecht schon sechs sechste sechsten sechster sechstes
sehr sei seid seien sein seine seinem seinen seiner seines seit seitdem selbst
selbst sich sie sieben siebente siebenten siebenter siebentes siebte siebten
siebter siebtes sind so solang solche solchem solchen solcher solches soll
sollen sollte sollten sondern sonst sowie später statt
tag tage tagen tat teil tel trotzdem tun
über überhaupt übrigens uhr um und uns unser unsere unserer unter
vergangene vergangenen viel viele vielem vielen vielleicht vier vierte vierten
vierter viertes vom von vor
wahr während währenddem währenddessen wann war wäre waren wart warum was wegen
weil weit weiter weitere weiteren weiteres welche welchem welchen welcher
welches wem wen wenig wenige weniger weniges wenigstens wenn wer werde werden
werdet wessen wie wieder will willst wir wird wirklich wirst wo wohl wollen
wollt wollte wollten worden wurde würde wurden würden
zehn zehnte zehnten zehnter zehntes zeit zu zuerst zugleich zum zunächst zur
zurück zusammen zwanzig zwar zwei zweite zweiten zweiter zweites zwischen
""".split()
)


# In[2]:


def to_txt(data, name):
    text = '\n'.join(data)
    print(len(text))
    Path('/mnt/data2/ptf/co_exp/' + name + '.txt').write_text(text)


# In[3]:


sents = pickle.load( open( "s_l.pkl", "rb" ) )


# In[9]:


def cl(x):
    return clean(x, no_urls=True, no_digits=True, no_punct=True, no_line_breaks=True, lang='de') 


# In[ ]:


sents = Parallel(n_jobs=4, backend='multiprocessing')(delayed(cl)(row) for row in tqdm(sents))


# In[4]:


len(sents)


# In[5]:


to_txt(sents, 'lemma')

pickle.dump( sents, open('s_l_cleaned.pkl', 'wb'))
# In[ ]:




