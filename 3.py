#!/usr/bin/python3
# -*- coding:utf8 -*-
import matplotlib.pyplot as plt  
from wordcloud import WordCloud  
import jieba 
from PIL import Image  
import numpy as np  
abel_mask = np.array(Image.open("./2.jpg"))   
text_from_file_with_apath = open('./report.txt',encoding="utf8").read()    
wordlist_after_jieba = jieba.cut(text_from_file_with_apath, cut_all = False)  
wl_space_split = " ".join(wordlist_after_jieba)    
my_wordcloud = WordCloud(width=1920,height=1080,max_font_size=160,
                min_font_size=32,background_color="white"
                ,mask=abel_mask
                     ).generate(wl_space_split) 
plt.imshow(my_wordcloud)
plt.axis("off")  
plt.show()