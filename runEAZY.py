import os
import gc
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import eazy.photoz
from scipy.integrate import cumulative_trapezoid
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import emcee
import eazy.filters
from scipy.optimize import curve_fit

#using eazy-py 0.8.7, whereas Bagley used 2015-05-08

from communitySurveySettings import surveyName, numPointings, eazyDir, surveys
dataDirectory = '/Users/lukeobrien/Desktop/TCD/SS/CapstoneProject/capstoneData.nosync/lightconeData/'

detectionFilter = "F158" #MUST BE HARD CODED DEPENDING ON WHAT BAND IS USED

params = {
    'CATALOG_FILE':   eazyDir + 'loadSort_toEAZY.cat', #NEED TO CHANGE CAT FILE EACH TIME
    'CATALOG_FORMAT': 'ascii.commented_header',      # text file, column names on the '#' line
    'FILTERS_RES':    eazyDir + 'FILTER.RES.roman',
    'TEMPLATES_FILE': '/Users/lukeobrien/Desktop/TCD/SS/CapstoneProject/EAZY/EAZY_LarssonTemplates.param',
    'MAIN_OUTPUT_FILE': eazyDir + 'loadSort_toEAZY', #NEED TO CHANGE EACH TIME

    'Z_MIN': 0.01, 'Z_MAX': 12.0, 'Z_STEP': 0.01,  #redshifts
    'PRIOR_ABZP': 23.9,                        #fluxes are in µJy
    'APPLY_PRIOR': 'n', #this makes high redhisft galaxies less likely for faint galaxies, Bagley turns it off as it states "the luminosity density of galaxies at high redshift is not yet fully understood”
    'MW_EBV': 0.0, 'CAT_HAS_EXTCORR': True,      #simulation: no Milky Way dust
    'SYS_ERR': 0.05,                           #extra 5% erro "to mimic systematic uncertainties expected with real data”
    'ADD_CGM':'n', #Circumgalactic Medium (CGM) absorption, it adds extra absorption around Lyman-α. Bagleys 2015 EAZY didn’t have it
        #eazypys default template error is a template error file with amplitude 0.20, the paper doesnt mention it.
    #'TEMP_ERR_A2': 0 can set it to zero here
    'Z_STEP_TYPE': 0 #if set to 1, the steps grow with redshift, 0.01 × (1 + z), which is about 0.08 at z = 7. Bagley used an even 0.01 everywhere, which is Z_STEP_TYPE = 0
}

if __name__ == '__main__':        # needed on macOS when EAZY uses several CPU cores

    #depth = float(input("Enter depth value: ")) #depth of survey
    #numPointings = int(input("Enter how many pointings: ")) from previous Bagley runs
    surveyArea = numPointings * 0.75 * 0.375 #deg^2

    #-------running EAZY on each chunk of the catalogue
    #the catalogue is split into chunk files by loadSort_toEAZY.py
    #the quantities the selection needs are saved for each chunk, chunks already saved are skipped so a stopped run can be restarted

    chunkFiles = sorted(glob.glob(eazyDir + 'loadSort_toEAZY_chunk*.cat'))
    print(f"{len(chunkFiles)} chunk files")
    startTime = time.time()

    for catFile in chunkFiles:
        resultsFile = catFile.replace('.cat', '_results.npz')
        if os.path.exists(resultsFile):
            print(f"{os.path.basename(catFile)} already done, skipping")
            continue

        params['CATALOG_FILE'] = catFile #replaces the CATALOG_FILE and MAIN_OUTPUT_FILE above for each chunk
        params['MAIN_OUTPUT_FILE'] = catFile.replace('.cat', '')

        ez = eazy.photoz.PhotoZ(param_file=None, translate_file=eazyDir + 'roman.translate',
                                zeropoint_file=None, params=params,load_prior=False, load_products=False, n_proc=8)

        ez.fit_catalog(ez.idx, n_proc=8, prior=False, beta_prior=False)   # the fitting
        ez.evaluate_zml(prior=False, beta_prior=False)                    # best redshift per galaxy

        ez.compute_lnp(prior=False, beta_prior=False)   #make sure P(z) has no prior, following Bagley
        zgrid = ez.zgrid                                #the redshift grid: 0.01, 0.02...12.00
        pz = np.exp(ez.lnp)    #P(z): one row per galaxy, one column per grid point. its already normalised

        
        ''' criterea
        (1) S/N ≥ 5 in detection band (H158)
        (2) at least 70% of P(z) is under the main peak
        (3) at least 35% of P(z) lies in 6.7 < z < 7.7
        (4) P(z) is higher in 6.7–7.7 than in any other integer-wide redshift range, so no galaxy is picked twice
        (5) P(z > 6) > 0.6
        (6) the best fit redshift is above zSample - 2
    
        '''

        #need the area of the distribution for most of these
        #so will use scipy to integrate the are of a fitted trapezoid

        cdf = cumulative_trapezoid(pz, zgrid, axis=1, initial=0) #this calculates the area under every curve

        def probabilityBelow(z):
            return cdf[:, np.searchsorted(zgrid, z)] #finds the position of z in the grid

        #(2)
        fractionUnderPeak = np.zeros(len(pz))

        for i, probability in enumerate(pz):
            maxPoint = probability.argmax()
            high = maxPoint
            low = maxPoint
            while high > 0 and probability[high] >= probability[high - 1]: #walking left
                high -= 1
            while low < len(probability) -1 and probability[low] >= probability[low + 1]: #walking right
                low += 1
            fractionUnderPeak[i] = cdf[i, low] - cdf[i, high]

        #(3) P(z) in every range lined up with the sample ranges
        rangeEdges = np.concatenate([[zgrid[0]], np.arange(0.7, zgrid[-1], 1.0), [zgrid[-1]]])   #0.01, 0.7, 1.7...11.7, 12
        
        #(4)
        pPerRange = np.diff(np.stack([probabilityBelow(edge) for edge in rangeEdges], axis=1), axis=1) #P in each range

        #(5)covers zSample - 1 for every sample
        integerZ = np.arange(1, 12)
        pBelowInteger = np.stack([probabilityBelow(z) for z in integerZ], axis=1)

        np.savez(resultsFile, id=ez.OBJID, zTrue=ez.ZSPEC, zBest=ez.zml, fractionUnderPeak=fractionUnderPeak,
                 rangeEdges=rangeEdges, pPerRange=pPerRange, integerZ=integerZ, pBelowInteger=pBelowInteger)
        print(f"{os.path.basename(catFile)}: {ez.NOBJ} galaxies saved ({(time.time() - startTime) / 60:.1f} min so far)", flush=True)

        del ez, pz, cdf #free the memory before the next chunk
        gc.collect()

    #----------------- loading the saved results of every chunk

    results = [np.load(f) for f in sorted(glob.glob(eazyDir + 'loadSort_toEAZY_chunk*_results.npz'))]
    zTrue = np.concatenate([r['zTrue'] for r in results])        # the simulation's true redshift (your z_spec column)
    zBest = np.concatenate([r['zBest'] for r in results])        #best fit redshift per galaxy
    zPhot = zBest                                       # EAZYs redshift
    fractionUnderPeak = np.concatenate([r['fractionUnderPeak'] for r in results])
    pPerRange = np.concatenate([r['pPerRange'] for r in results])
    pBelowInteger = np.concatenate([r['pBelowInteger'] for r in results])
    rangeEdges, integerZ = results[0]['rangeEdges'], list(results[0]['integerZ'])
    galaxyId = np.concatenate([r['id'] for r in results])            #catalogue id of each galaxy
    print(f"\n{len(zTrue)} galaxies with EAZY results")

    #----------------- computing and plotting after EAZY redshifts

    #the four redshift samples: 5.7-6.7, 6.7-7.7, 7.7-8.7 and 8.7-9.7
    summary = [] #one row per sample, saved to selectionSummary.txt
    selectedPerSample = {} #which galaxies each sample selects, saved to selectedGalaxies.npz

    for zSample in [2,3,4,5,6, 7, 8, 9]:
        zLow, zHigh = zSample - 0.3, zSample + 0.7   #6.7 < z < 7.7 (ranges centred on 6.2, 7.2, 8.2, 9.2)


        #(2)
        criteria2 = fractionUnderPeak >= 0.7

        sampleRange = np.argmin(np.abs(rangeEdges[:-1] - zLow))       #which range is 6.7-7.7

        #(3)
        '''
        for i, probability in enumerate(seclectedGal):
            while 6.7 < pz < 7.7:
                if cdf[i,probability] >= 0.35:
        '''
        criteria3 = pPerRange[:, sampleRange]

        #(4) 
        criteria4 = pPerRange.argmax(axis=1) == sampleRange

        #(5)
        criteria5 = 1 - pBelowInteger[:, integerZ.index(zSample - 1)]

        #(6)
        criteria6 = (zBest > zSample - 2)


        selected = (fractionUnderPeak >= 0.7) & (criteria3 >= 0.35) & criteria4 & (criteria5 > 0.6) & criteria6

        #checking, using Bagley's definitions
        zMid = (zLow + zHigh) / 2    #7.2
        contaminant = selected & (np.abs(zTrue - zMid) > 1)
        trueInRange = (zTrue >= zLow) & (zTrue < zHigh) #Bagley, 5.7 <= z < 6.7, every galaxy here already has S/N >= 5
        recovered = selected & trueInRange
        recoveryFraction = recovered.sum() / max(trueInRange.sum(), 1) #selected with true z in range / all S/N >= 5 with true z in range
        contaminationFraction = contaminant.sum() / max(selected.sum(), 1) #selected contaminants / all selected
        print(f"\nz = {zSample} sample ({zLow} <= z < {zHigh})")
        print(f"Selected: {selected.sum()}, contaminants: {contaminant.sum()} ({contaminationFraction:.1%})")
        print(f"Recovered: {recovered.sum()} of {trueInRange.sum()} true {zLow}<=z<{zHigh} galaxies ({recoveryFraction:.1%})")

        summary.append([zSample, zLow, zHigh, selected.sum(), recovered.sum(), trueInRange.sum(), recoveryFraction,
                        contaminant.sum(), contaminationFraction])
        selectedPerSample[f'selected_z{zSample}'] = selected #each mask is named selected_z*


    #saving the numbers for every sample, and which galaxies were selected (for the luminosity functions later)
    np.savetxt(eazyDir + 'selectionSummary.txt', np.array(summary), fmt=['%d', '%.1f', '%.1f', '%d', '%d', '%d', '%.4f', '%d', '%.4f'],
               header='zSample zLow zHigh nSelected nRecovered nTrueInRange recoveryFraction nContaminants contaminationFraction')
    np.savez(eazyDir + 'selectedGalaxies.npz', id=galaxyId, zTrue=zTrue, zBest=zBest, **selectedPerSample)
    print(f"\nsaved {eazyDir}selectionSummary.txt and {eazyDir}selectedGalaxies.npz")


    #---------------- mag vs EAZY redshift
    
    catalogueFluxes = []
    catalogueIds = []
    for catFile in sorted(glob.glob(eazyDir + 'loadSort_toEAZY_chunk*.cat')):
        with open(catFile) as file:
            columnNames = file.readline().lstrip('#').split() #takes the header line
        
        chunk = pd.read_csv(catFile, sep=r'\s+', comment='#', header=None, names=columnNames, usecols=['id', 'f_F158']) #change depending on what band I want
        catalogueIds.append(chunk['id'].to_numpy())
        catalogueFluxes.append(chunk['f_F158'].to_numpy())
    
    fluxBand = np.concatenate(catalogueFluxes) #scattered band flux in microJansky, same order as the EAZY results
    
    magBand = 23.9 - 2.5 * np.log10(fluxBand) #AB magnitude, zero point 23.9 for microJansky
    
    #sort galaxoes into recovered and contaminated groups
    recoveredAll = np.zeros(len(zBest), dtype=bool)
    contaminatedAll = np.zeros(len(zBest), dtype=bool)
    
    sampleEdges = [ 1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7, 9.7] 
    sampleMids = [ 2.2, 3.2, 4.2,5.2, 6.2, 7.2, 8.2, 9.2]
    initialSample = 2
    
    for name, mask in selectedPerSample.items():
        
        zLow, zHigh = initialSample - 0.3, initialSample + 0.7 
        zMid = (zLow + zHigh) / 2
        inRange = (zTrue >= zLow) & (zTrue < zHigh)
        outRange = np.abs(zTrue - zMid) > 1 #using true z to test if the gal is actually outside the range
        
        recoveredAll = recoveredAll | (mask & inRange)
        contaminatedAll = contaminatedAll | (mask & outRange)
        
        
        #plt.scatter(zBest[mask],magBand[mask], s= 2, label=f'z = {initialSample} sample')
        initialSample += 1
    
    plt.scatter(zBest[recoveredAll], magBand[recoveredAll], s=2, color='#3a7dc0',
                label="Recovered: Galaxy's true z is within sample range ")
    plt.scatter(zBest[contaminatedAll], magBand[contaminatedAll], s=2, color='#d0343d',
                label="Contaminants: Galaxy's true z is > 1 from range's midpoint")
    
    ax = plt.gca() #the current axes, needed for minor ticks
    ax.set_xticks(sampleEdges) #major ticks on the range edges
    ax.set_xticks(sampleMids, minor=True) #minor ticks on the midpoints

    ax.grid(which='minor', axis='x', linestyle='--') #midpoints
    ax.grid(which='major', axis='x', linestyle='--', color='0.3') #range edges
    ax.grid(which='major', axis='y', linestyle='--') #magnitude gridlines
    
    plt.legend(loc='lower right')
    #plt.yscale('log') #fluxes if plotting flux on y axis as flux span a factor of 20
    plt.xlabel("EAZY Redshift")
    plt.ylabel("F158 mag")
    plt.title(f"{surveyName}, {numPointings} Roman pointings, F158 5σ depth {surveys[surveyName]['F158']}")
    plt.show()
    
    #--------F158 mag vs true redshift with colour bar of recovery fraction
    #recovery fraction = detected galaxy recovered bin / all detected galaxies bin  
    
    inSamples = (zTrue >= 1.7) & (zTrue < 9.7) #galaxies with true z in one of the four sample ranges
    
    #bins of true redshift and observed F158 magnitude
    zBins = np.arange( 1.7, 9.7 + 0.05 , 0.1) #the +0.05 is to make sure the last bin up to 9.7 is used
    magBins = np.arange(np.floor(magBand[inSamples].min()), np.ceil(magBand[inSamples].max()) , 0.2) #ceil of the scalar x is the smallest integer i, such that i >= x

    totalCounts, _, _ = np.histogram2d(zTrue, magBand, bins=[zBins, magBins]) #all detected galaxies per bin
    recoveredCounts, _, _ = np.histogram2d(zTrue[recoveredAll], magBand[recoveredAll], bins=[zBins, magBins]) #recovered per bin

    recoveryFraction = np.full(totalCounts.shape, np.nan) #empty bins stay NaN
    recoveryFraction = recoveredCounts / totalCounts

    ax = plt.gca()
    ax.set_xticks(sampleEdges)
    ax.set_xticks(sampleMids, minor=True)

    ax.grid(which='minor', axis='x', linestyle='--') #midpoints
    ax.grid(which='major', axis='x', linestyle='--', color='0.3') #range edges
    ax.grid(which='major', axis='y', linestyle='--') #magnitude gridlines

    plt.pcolormesh(zBins, magBins, recoveryFraction.T, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='Recovery fraction')
    plt.xlabel('True redshift')
    plt.ylabel('F158 (mag)')
    plt.title(f"{surveyName}, {numPointings} Roman pointings, F158 5σ depth {surveys[surveyName]['F158']}")
    plt.show()
    
    #--------plot F158 mag vs true redshift with colour bar of contamination fraction
    #contamination fraction = detected galaxy contaminated bin / all detected galaxies bin  
    
    inSamples = (zBest >= 1.7) & (zBest < 9.7) #galaxies with true z in one of the four sample ranges
    
    #bins of true redshift and observed F158 magnitude
    zBins = np.arange(1.7, 9.7 + 0.05, 0.1)
    magBins = np.arange(np.floor(magBand[inSamples].min()), np.ceil(magBand[inSamples].max()) , 0.2) #ceil of the scalar x is the smallest integer i, such that i >= x

    selectedAll = recoveredAll | contaminatedAll

    totalCounts, _, _ = np.histogram2d(zBest[selectedAll], magBand[selectedAll], bins=[zBins, magBins]) #all detected galaxies per bin
    contaminatedCounts, _, _ = np.histogram2d(zBest[contaminatedAll], magBand[contaminatedAll], bins=[zBins, magBins]) #recovered per bin

    contaminatedFraction = np.full(totalCounts.shape, np.nan) #empty bins stay NaN
    contaminatedFraction = contaminatedCounts / totalCounts

    ax = plt.gca()
    ax.set_xticks(sampleEdges)
    ax.set_xticks(sampleMids, minor=True)

    ax.grid(which='minor', axis='x', linestyle='--') #midpoints
    ax.grid(which='major', axis='x', linestyle='--', color='0.3') #range edges
    ax.grid(which='major', axis='y', linestyle='--') #magnitude gridlines

    from matplotlib.colors import ListedColormap #used this to make the 'creamish' side of the colour bar darker so its more visible
    creamToPurple = ListedColormap(plt.cm.magma_r(np.linspace(0.1, 1, 256))) #(A,B,C). A is the light colour, B is the darker colour, C is the number of evenly spaced colours

    plt.pcolormesh(zBins, magBins, contaminatedFraction.T, cmap=creamToPurple, vmin=0, vmax=1)
    plt.colorbar(label='Contamination fraction')
    plt.xlabel('EAZY redshift')
    plt.ylabel('F158 (mag)')
    plt.title(f"{surveyName}, {numPointings} Roman pointings, F158 5σ depth {surveys[surveyName]['F158']}")
    plt.show()
    
    
    
    






'''
    #--------EAZY redshift vs true redshift

    dz = (zPhot - zTrue) / (1 + zTrue) #dividing by (1 + z) puts every galaxy on a fair scale
    nmad = (np.median(np.abs(dz - np.median(dz))))/0.6745 #typical scatter (Normalised Median Absolute Deviation), this deals with outliers
    outliers = np.mean(np.abs(dz) > 0.15)  #fraction badly wrong
    print(f'sigma_NMAD = {nmad:.3f},  outlier fraction = {100*outliers:.1f}%')

    plt.figure(figsize=(6, 6))
    plt.scatter(zTrue, zPhot, s=2, alpha=0.4, color='k')
    plt.plot([0, 12], [0, 12], 'r-') #perfect answer line
    numGal = len(zTrue)
    plt.title(f"{numGal} galaxies detected in {detectionFilter}")
    plt.xlabel('True redshift')
    plt.ylabel('EAZY redshift')
    plt.xlim(0, 10); plt.ylim(0, 12)
    plt.savefig(eazyDir + 'zphot_vs_ztrue.png', dpi=150)
    plt.show()
'''


'''
points on the red line are galaxies EAZY got right.
points far from the line are catastrophic failures. The usual one is a galaxy at z ~ 1–2 that EAZY thinks is at z ~ 6+, or the visa versa, 
because the breaks look alike to ERAZY (alpha lyman and balmer break)
'''
