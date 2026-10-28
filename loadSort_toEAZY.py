#EAZY for calcualting redshifts within lightcone_nc.dat

# download libraries and templates
import eazy
import os
import glob
import eazy.utils
import eazy.photoz
import numpy as np
import pandas as pd
import sys
import time 

print('eazy.utils.DATA_PATH = [...]/' + '/'.join(eazy.utils.DATA_PATH.split('/')[3:]))

if not os.path.exists(eazy.utils.DATA_PATH):
    eazy.fetch_eazy_photoz()


print(time.ctime() + '\n')

print(sys.version + '\n')

dataDirectory = '/Users/lukeobrien/Desktop/TCD/SS/CapstoneProject/capstoneData.nosync/lightconeData/'
outputFile = '/Users/lukeobrien/Desktop/TCD/SS/CapstoneProject/plots/photoRedshiftsFirst.png'

#need to save in chunks for EAZY software

sys.path.append('/Users/lukeobrien/Desktop/TCD/SS/CapstoneProject/EAZY/') #surveySettings.py lives there
from communitySurveySettings import surveys, surveyName, numPointings, eazyDir
os.makedirs(eazyDir, exist_ok=True) #makes the survey's folder if it doesn't exist yet

testSampleSize = 10000 #10000 for a quick test, None to use every detected galaxy
chunkSize = 150000 #galaxies per EAZY run



#column numbers

#F146 and F213 arent included in Bagley paper, they say F213 isnt included in the SAM catalog. 
#however it is in the catalog im using, so I will include it

filterColumns = {'F062': 10, 'F087': 11, 'F106': 12, 'F129': 13, 'F146': 14, 'F158': 15, 'F184': 16, 'F213': 17} #every Roman filter in the lightcone
depths = surveys[surveyName]
filters = {name: filterColumns[name] for name in depths} #only does the chosen surveys filters


cols = [4] + [5] + [6] + list(filters.values())     #true redshift, and the 8 filters



'''
0: "halo_id_nbody",
1: "gal_id",
2: "gal_type",
3: "z_nopec",
4: "redshift",
5: "ra",
6: "dec",
7: "UV1500_rest_dust",
8: "UV2300_rest_dust",
9: "UV2800_rest_dust",
10: "Roman_F062_dust",
11: "Roman_F087_dust",
12: "Roman_F106_dust",
13: "Roman_F129_dust",
14: "Roman_F146_dust",
15: "Roman_F158_dust",
16: "Roman_F184_dust",
17: "Roman_F213_dust",
'''


#zRange = 0.01 < z < 12
dz = 0.01
zRange = (0.01,12)



ZP = 23.9 #zero point
'''
depth = float(input("Enter depth value: ")) #depth of survey
dropoutFilterDepth = depth + 0.5
sigma = (10**(-((depth-ZP))/2.5))/5 #divided by 5, as it is a 5sigma depth. (micro Janksy)
dropoutSigma = (10**(-((dropoutFilterDepth-ZP))/2.5))/5
'''
#meaning a galaxy at 29mag has a signal to noise ratio of 5

filterSigma = {name: (10**(-(depths[name] - ZP)/2.5))/5 for name in filters} #a sigma value from each filters 5 sigma depth (in micro Jansky)

#filterSigma = {name: (dropoutSigma if name in dropoutFilters else sigma) for name in filters} from Bagley runs

#----------load data in

files = sorted(glob.glob(os.path.join(dataDirectory, 'lightcone_nc*.dat')))


allValues = []

for file in files:
    df = pd.read_csv(file,sep=r'\s+', comment='#', header=None, usecols=cols)
    allValues.append(df)
    print(f"{os.path.basename(file):25s} {len(df):8d} galaxies")

data = pd.concat(allValues, ignore_index=True)

#------used this code from my lumFuncVSmag_BagleyFig2replicate script
#its used to only consider galaxies withoin 1 roman pointing (through 1 box)

z = data[4].to_numpy()
ra = data[5].to_numpy() #ra
dec = data[6].to_numpy() #dec

ra = np.where(ra > 180, ra - 360, ra) #this renames 359.3 for example as -0.7


#----------lightcone using ra and dec 

#flat sky coordinates
inBin = (z > 6.7) & (z < 7.7)
dec0 = 0.5 * (dec[inBin].min() + dec[inBin].max()) #using Dec0 for whole sky is a flat sky approximation, ok for small angles (1.4 degrees in this case)
x = ra * np.cos(np.radians(dec0)) #true degrees on the sky at Dec at centre of field
y = dec

xMin, yMin = x[inBin].min(), y[inBin].min()

#Roman fields; three 0.75 x 0.375 deg fields stacked, one rotated into the leftover strip
romanWidth, romanHeight = 0.75, 0.375

#tuple:        left,          right,         bottom,                     top
#romanBox = (x >= xMin) & (x < xMin + romanWidth) & (y >= yMin) & (y < yMin + romanHeight) #for 1 pointing


romanBox = (x >= xMin) & (x < xMin + romanWidth) & (y >= yMin) & (y < yMin + numPointings * romanHeight)

data = data[romanBox].reset_index(drop=True)
print(f"{len(data)} galaxies inside the Roman field 0.75 x({numPointings * romanHeight} deg)")



allFluxes = {}

for name, col in filters.items():
    allFluxes[name] = 10**(-(data[col].to_numpy() - ZP)/2.5)


rng = np.random.default_rng(50) #constant seed  makes it repeatable

scatteredFluxes = {}
for name in filters:
    specificSigma = filterSigma[name]
    noise = rng.normal(0, specificSigma, len(data)) + (0.05 * specificSigma * rng.standard_cauchy(len(data)) ) 
    #0.05 * sigma, this is because the half width of Lorenztian is 0.05sigma
    #this fits a voigt profile, (Gaussian core + Lorentzian wings).
    scatteredFluxes[name] = allFluxes[name] + noise #voigt profile used for scatter

detected = scatteredFluxes['F158'] / filterSigma['F158'] >= 5  #S/N >= 5 in F158
detectedIndex = np.flatnonzero(detected) #row numbers of detected galaxies
print(f"\n{len(detectedIndex)} of {len(data)} galaxies detected with a S/N >= 5\n")

if testSampleSize is not None: #a random sample for quick tests, None for the full run
    detectedIndex = np.sort(rng.choice(detectedIndex, size=testSampleSize, replace=False))  #random sample for testing

trueRedshift = data[4].to_numpy()[detectedIndex] #true redshift



#-------write file for EAZY

numGal = len(detectedIndex)
names = ['id']
columns = [np.arange(1, numGal + 1)]

for name in filters:
    names += [f'f_{name}', f'e_{name}']
    columns += [scatteredFluxes[name][detectedIndex], np.full(numGal, filterSigma[name])] #flux and then its error

names += ['z_spec']
columns += [trueRedshift]

table = np.column_stack(columns) #put columns side by side
formats = ['%d'] + ['%.6e'] * (2 * len(filters)) + ['%.4f'] #this extra column is only needed if i extract true redshift

####################
#----------write the catalogue in chunks, EAZY needs ~24 KB of memory per galaxy so 150,000 is ~3.6 GB
#old chunks and their results are removed first, so results never get mixed with a different catalogue
for oldFile in glob.glob(eazyDir + 'loadSort_toEAZY_chunk*'):
    os.remove(oldFile)

numChunks = 0
for start in range(0, numGal, chunkSize):
    chunkFile = eazyDir + f'loadSort_toEAZY_chunk{numChunks:02d}.cat'
    np.savetxt(chunkFile, table[start:start + chunkSize], header=' '.join(names), fmt=formats) #id keeps running across chunks
    numChunks += 1
print(f"wrote {numGal} galaxies to {numChunks} chunk files of up to {chunkSize} galaxies")
####################

#--------now setting EAZY up


import eazy.filters

sedpyDir = '/opt/anaconda3/envs/EAZY/lib/python3.12/site-packages/sedpy/data/filters/'

filterResponseCurve = eazy.filters.FilterFile('FILTER.RES.latest') #EAZYs 424 filters
filterNumber = {}

#this adds the Roman filters above to the end of the list of filters already existing within EAZY
for name in filters:
    curve = np.loadtxt(sedpyDir + f'roman_wfi_{name.lower()}.par', usecols=(0, 1))
    wave, lightThrough = curve[:, 0], curve[:, 1]
    keep = lightThrough > 0 #drop the zero parts
    newFilter = eazy.filters.FilterDefinition(wave=wave[keep], throughput=lightThrough[keep] / lightThrough.max()) #lightThrough / lightThrough.max scales each curve to 1
    newFilter.name = f'Roman_{name} lambda_c= {newFilter.pivot:.4e}' #EAZY needs 'lambda_c' in the name
    filterResponseCurve.filters.append(newFilter)
    filterNumber[name] = len(filterResponseCurve.filters) #its number in the list
    print(name, '-> filter number', filterNumber[name])

filterResponseCurve.write(file=eazyDir + 'FILTER.RES.roman', verbose=False)

#linking my column names to the filter names
with open(eazyDir + 'roman.translate', 'w') as fp:
    for name in filters:
        fp.write(f'f_{name} F{filterNumber[name]}\n') #flux column
        fp.write(f'e_{name} E{filterNumber[name]}\n') #error column


#a problem is that the simulation made its magnitudes with its own Roman curves. These sedpy curves are NASA's newer 2024 version, so may be different


