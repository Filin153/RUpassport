# RUpassport #

## What is this? ##
Library to recognizes data from Russian passports and returns them

## Quick Guide ##
    pip install RUpassport
####
    from RUpassport import Pasport
    
    p = Pasport()
    pasport_info = p.recognize_pasport("img.png", "file", "123")
    print(pasport_info)
