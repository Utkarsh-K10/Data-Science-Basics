#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:58:00 2020

@author: utkarshkushwaha
"""

import ssl
import os
os.chdir('/Users/utkarshkushwaha/Desktop/Spyderworkspace/DataScience_rvidon')
import lxml
import pandas as pd
import numpy as np
import urllib.request
import re
from bs4 import BeautifulSoup

#creating source
#Collecting  data in source
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


url = 'https://www.srvmedia.com/Digital-Marketing-faq'
digital_market_source = urllib.request.urlopen(url).read()

digital_market_source


# parsing and creating beautifful soupe


soup = BeautifulSoup(digital_market_source,'html.parser')
soup


text = ""
for paragraph in soup.find_all('p'):
    print(paragraph.text)
    text += paragraph.text 
    text += '\n\n'
    
len(text)
answer = []

answer = re.sub(r'\[[0-9]*\]','',text)
answer = re.sub(r'\s+','',answer)