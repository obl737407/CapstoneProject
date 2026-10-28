
#survey settings 
#read by loadSort_toEAZY.py, runEAZY.py (and plotSEDs_PDF.py through runEAZY)

surveyName = 'RXDF' #this is changed to suit whatever survey  is needed to be simulated
numPointings = 1

#5 sigma point source depths for each filter (AB) 
#HLWAS community survey, settings got from https://roman-docs.stsci.edu/roman-community-defined-surveys/high-latitude-wide-area-survey
surveys = {
    'HLWAS_Medium':    {'F106': 26.5, 'F129': 26.4, 'F158': 26.4},
    'HLWAS_Deep':      {'F087': 27.7, 'F106': 27.7, 'F129': 27.6, 'F146': 28.3, 'F158': 27.5, 'F184': 27.0, 'F213': 25.9},
    'HLWAS_UltraDeep': {'F106': 28.2, 'F129': 28.2, 'F158': 28.1},
    'RXDF':            {'F062': 30.0, 'F087': 30.0, 'F106': 30.0, 'F129': 30.0, 'F158': 30.0, 'F184': 29.0, 'F213': 28.0},
    'Bagley_29.38':    {'F062': 29.88, 'F087': 29.88, 'F106': 29.38, 'F129': 29.38, 'F158': 29.38, 'F184': 29.38, 'F213': 29.38}, #my earlier runs using Bagley settings
    #need to test this more 
    'HLWAS_UltraDeep_andDeep': {'F087': 27.7, 'F106': 27.7, 'F129': 27.6, 'F146': 28.3, 'F158': 27.5, 'F184': 27.0, 'F213': 25.9, 'F106': 28.2, 'F129': 28.2, 'F158': 28.1}

}


eazyDir = f'/Users/lukeobrien/Desktop/TCD/SS/CapstoneProject/EAZY/{surveyName}_{numPointings}pointings/'

