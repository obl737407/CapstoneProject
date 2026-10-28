#SED examples of recovered and contaminant galaxies within each reshift range
#reuses the EAZY settings and run folder from runEAZY.py  and its results
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eazy.photoz
import eazy.filters
from runEAZY import params, eazyDir #same EAZY settings


sampleRanges = [2,3,4,5,6,7,8,9]
chunkSize = 150000
rng = np.random.default_rng(100) 
lymanAlpha = 1215.67 #Angstroms

if __name__ == '__main__':

    #---runEAZY.py saves selection redshift and the quantities from each criteria
    selection = np.load(eazyDir + 'selectedGalaxies.npz')
    galaxyId, zTrue, zBest = selection['id'], selection['zTrue'], selection['zBest'], 


    results = [np.load(f) for f in sorted(glob.glob(eazyDir + 'loadSort_toEAZY_chunk*_results.npz'))]
    fractionUnderPeak = np.concatenate([r['fractionUnderPeak'] for r in results])
    pPerRange = np.concatenate([r['pPerRange'] for r in results])
    pBelowInteger = np.concatenate([r['pBelowInteger'] for r in results])
    rangeEdges, integerZ = results[0]['rangeEdges'], list(results[0]['integerZ'])

    #F158 magnitude of every galaxy from the chunks
    magF158 = []  
    
    for catFile in sorted(glob.glob(eazyDir + 'loadSort_toEAZY_chunk*.cat')):
        with open(catFile) as file:
            columnNames = file.readline().lstrip('#').split()
        chunk = pd.read_csv(catFile, sep=r'\s+', comment='#', header=None, names=columnNames, usecols=['f_F158'])
        magF158.append(23.9 - 2.5 * np.log10(chunk['f_F158'].to_numpy()))
    magF158 = np.concatenate(magF158)

    fits = {}
    templateGrid = None   

    for sample in sampleRanges:
        
        selected = selection[f'selected_z{sample}']
        
        #passing criteria from runEAZY.py
        zLow, zHigh = sample - 0.3, sample + 0.7
        zMid = (zLow + zHigh) / 2
        sampleRange = np.argmin(np.abs(rangeEdges[:-1] - zLow))
        criteria = {'(2) peak >= 70%': fractionUnderPeak >= 0.7,
                    '(3) P(range) >= 35%': pPerRange[:, sampleRange] >= 0.35,
                    '(4) most P in range': pPerRange.argmax(axis=1) == sampleRange,
                    f'(5) P(z>{sample - 1}) > 0.6': (1 - pBelowInteger[:, integerZ.index(sample - 1)]) > 0.6,
                    f'(6) zBest > {sample - 2}': zBest > sample - 2}
    
        #galaxy exmaples
        
        trueInRange = (zTrue >= zLow) & (zTrue < zHigh)
        
        groups = {f'Bright galaxy recovered within z~{sample} range': selected & trueInRange & (magF158 < 28),
                  f'Faint galaxy recovered within z~{sample} range': selected & trueInRange & (magF158 > 29),
                  f'Contaminant (true z > 1 from z~{sample} midpoint)': selected & (np.abs(zTrue - zMid) > 1)}
        examples = {name: rng.choice(np.flatnonzero(mask)) for name, mask in groups.items() if mask.any()}
        for name, mask in groups.items():
            print(f"{name}: {mask.sum()} galaxies" + ("" if mask.any() else " (none, skipped)"))
    
        
        #-------plot the example galaxies

        for name, i in examples.items():
            chunkNumber = (galaxyId[i] - 1) // chunkSize
            if chunkNumber not in fits:
                params['CATALOG_FILE'] = eazyDir + f'loadSort_toEAZY_chunk{chunkNumber:02d}.cat'
                params['MAIN_OUTPUT_FILE'] = eazyDir + f'sedExamples_chunk{chunkNumber:02d}'
                fits[chunkNumber] = eazy.photoz.PhotoZ(param_file=None, translate_file=eazyDir + 'roman.translate', zeropoint_file=None,
                                                           params=params, load_prior=False, load_products=False, n_proc=0)
            
            ez = fits[chunkNumber]
            rows = np.array([(galaxyId[j] - 1) % chunkSize for j in examples.values() if (galaxyId[j] - 1) // chunkSize == chunkNumber])
            ez.fit_catalog(rows, n_proc=0, prior=False, beta_prior=False)
            ez.evaluate_zml(prior=False, beta_prior=False)



        for name, i in examples.items():
            ez = fits[(galaxyId[i] - 1) // chunkSize]
            best = ez.show_fit(galaxyId[i], get_spec=True, show_fnu=1) #fit at EAZYs best redshift, get_spec returns the dictionary of EAZYs key names
            atTrue = ez.show_fit(galaxyId[i], get_spec=True, show_fnu=1, zshow=zTrue[i]) #best template forced to the true redshift
            galaxyRowInChunk = (galaxyId[i] - 1) % chunkSize #this galaxys row in its chunk
    
            inRange = (best['templz'] > 4000) & (best['templz'] < 25000) #Roman filters only in Angstroms
            inRangeTrue = (atTrue['templz'] > 4000) & (atTrue['templz'] < 25000) #for mapping the x axis

            #pivot - each filters pivot wavelength (7 of them)
            #fobs - the measured flux in each filter from my data (7 of them)
            #model - the fitted flux in each filter, the average template spectrum through each filter curve (7 of them)
            #templf - flux of the template spectrum at each of those wavelengths (about 6000)
            #templz - wavelengths of the template spectrum ( about 6000)

            #SED at EAZY redshift
            plt.figure()
            #best fit template at EAZY redhisft
            plt.plot(best['templz'][inRange]/1e4, best['templf'][inRange], color = 'black', label = f"Template at EAZY z ($\\chi^2$={best['chi2']:.1f})") #template spectrum at EAZY z
            #the pivot key is each filter pivot wavelength,
            #observed photometry
            plt.errorbar(best['pivot']/1e4, best['fobs'], yerr = best['efobs'], fmt='o', color = 'green', label = 'Observed photometry')  #so this plots the best fit SED for the pivot wavelength of each filter
            #model photometry
            plt.scatter(best['pivot'] / 1e4, best['model'], color = 'blue',zorder=3, label = f'Model photometry at EAZY z~ {zBest[i]:.1f}')
            '''
            for f in ez.filters:
                plt.plot(f.wave/1e4, f.throughput/(f.throughput.max()*150), alpha=0.5)
                #plt.plot(f.pivot/1e4, 1.05, f.name.split()[0], ha='center', fontsize=8)
            '''
            filterColours = plt.cm.rainbow(np.linspace(0, 1, len(ez.filters)))

            for filt, colour in zip(ez.filters, filterColours):
                plt.plot(filt.wave/1e4, filt.throughput/(filt.throughput.max()*150), color=colour, alpha=0.5)

            
            plt.legend(loc = 'upper left')
            plt.xlabel(r"$\lambda_{\rm obs}$ ($\mu$m)")
            plt.ylabel(r"F$_\nu$ ($\mu$Jy)")
            #plt.title(f"SED of galaxy {galaxyId[i]} at z~{sample}")
            plt.title(f"SED of {name} at EAZY z, ID: {galaxyId[i]}")
            plt.show()

            #SED at True redshift
            
            #best fit template at true redhisft
            plt.plot(atTrue['templz'][inRangeTrue]/1e4, atTrue['templf'][inRangeTrue],color = 'black' ,label = f"Template at True z ($\\chi^2$={best['chi2']:.1f})") #template spectrum at the true z
            #model photometry
            plt.scatter(atTrue['pivot'] / 1e4, atTrue['model'], color = 'red',zorder=3, label = f'Model photometry at True z~ {zTrue[i]:.1f}')
            #observed photometry
            plt.errorbar(best['pivot']/1e4, best['fobs'], yerr = best['efobs'], fmt='o', color = 'green',label = 'Observed photometry')  #so this plots the best fit SED for the pivot wavelength of each filter
            
            for filt, colour in zip(ez.filters, filterColours):
                plt.plot(filt.wave/1e4, filt.throughput/(filt.throughput.max()*150), color=colour, alpha=0.5)
            
            plt.legend(loc = 'upper left')
            plt.xlabel(r"$\lambda_{\rm obs}$ ($\mu$m)")
            plt.ylabel(r"F$_\nu$ ($\mu$Jy)")
            #plt.title(f"SED of galaxy {galaxyId[i]} at z~{sample}")
            plt.title(f"SED of {name} at True z, ID: {galaxyId[i]}")
            plt.show()

            #PDF
            plt.figure()
            plt.plot(ez.zgrid, np.exp(ez.lnp[galaxyRowInChunk]), color = 'black') #redshift probability distribution
            plt.axvline(zTrue[i], color='red', label='True z') #true redshift
            plt.axvline(zBest[i], color='blue', ls='--', label='EAZY z') #EAZY redshift
            plt.legend(loc = 'upper right')
            plt.xlabel("Redshift")
            plt.ylabel("P(z)")
            plt.title(f"PDF for {name}, ID: {galaxyId[i]}")
            plt.show()


    print("\n",best.keys(),"\n") #prints keys



'''
plt.legend(loc='lower right')
#plt.yscale('log') #fluxes if plotting flux on y axis as flux span a factor of 20
plt.xlabel("EAZY Redshift")
plt.ylabel("F158 mag")
plt.title("f{numPointings} Roman pointings (7 filters, F158 5σ depth at 29.38)")
plt.show()
'''