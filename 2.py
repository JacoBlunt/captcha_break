# -*- coding: utf-8 -*-
import os 
def getSubDirs(rootDir): 
    for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        if os.path.isdir(path): 
            print(path)


print('=======================================')
Test2('E:\\git_work\\captcha_break\\pic_src\\') 