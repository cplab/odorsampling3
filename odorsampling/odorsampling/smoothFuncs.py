#Mitchell Gronowitz
#Running dPsiSaturation with separate excel docs and graph functions


import time
import multiprocessing

from odorsampling.RnO import (
    QSpace, Epithelium, Odorscene, Ligand, dPsiBarSaturation, colorMapSumOfSquares,
    graphFromExcel, dPsiGraphFromExcel, dPsiOccActGraphFromExcel
)
from odorsampling import config, utils


def testdPsiBarSat(fixed, aff_sd=[0.5,1.5], eff_sd=[0.05,1.0], numRecs=30, c=1, dim=2, qspaces=[4,10,30], purpose="standard"):
    """Runs multiple graphs of given qspaces at one time
    Optional - run makeSimilar (line 25), to create 3 epitheliums with equal eff and aff SD's (only rec means differ)
    Otherwise - make sure there are three saved epithelium files with correct names

    Returns a dPsiBar graph with three qspaces, an act and occ graph, and excel docs with details
    
    fixed = True if want eff = 1
    c = convergence ratio of recs to glom
    purpose = reason for running simulation = either "eff", "aff", "c", "recs", "redAff", "dim", or 'standard'"""
    
    #Run this function if don't already have saved epithelium files to use
    makeSimilar(numRecs, aff_sd, eff_sd, purpose, qspaces, dim)
    
    startTime = time.time()
    
    purp = purpFunction(purpose, aff_sd, eff_sd, numRecs, c, dim)
    
    pdfName = "LigandSat with varying qspaces" + purp
    plotTitle = "Saturation of dPsiBar" + purp
    
    #i = 0
    labelNames = []
    excelNames = []
    end = False
    
    
    if __name__ == '__main__':
        jobs = []
    
    
        #while i < len(qspaces):
        for i, qspacesItem in enumerate(qspaces):    
            space = []
            #j = 0
            #while j < dim:
            for j in range(dim):    
                space.append((0,qspacesItem))
                #j += 1
            qspace = QSpace(space)
            epith = Epithelium.load("1. SavedEpi_" + str(qspace.size[0]) + purp + ".csv")

            labelNames.append(str(qspace.size[0]) + " qspace")
            excelNames.append("LigandSat with " + str(qspace.size[0]) + " qspace" + purp)
        
            if i == (len(qspaces) - 1):
                end = True
          
            p = multiprocessing.Process(target=dPsiBarSaturation, args=(epith, .01, qspace, pdfName, labelNames[i], excelNames[i], fixed ,c, plotTitle, end, purp, False))
            jobs.append(p)
            p.start()
        
        
            #epi, dn, qspace, pdfName, labelName, excelName, fixed eff
            #dPsiBarSaturation(epith, .01, qspace, pdfName, labelNames[i], excelNames[i], fixed ,c, plotTitle, end, purp, graphIt=False)
        
            #i += 1
        
        
        for j, job in enumerate(jobs):
            job.join()
        
            
def testdPsiBarSatColorMap(fixed, aff_sd = [0.5, 1.5], eff_sd = [0.05, 1.0], numRecs = 30, c = 1, dim = 2, qspaces = [4,10,30], purpose = "standard", qunits = 3):
    """Runs multiple graphs of given qspaces at one time
    Optional - run makeSimilar (line 25), to create 3 epitheliums with equal eff and aff SD's (only rec means differ)
    Otherwise - make sure there are three saved epithelium files with correct names

    Returns a dPsiBar graph with three qspaces, an act and occ graph, and excel docs with details
    
    fixed = True if want eff = 1
    c = convergence ratio of recs to glom
    purpose = reason for running simulation = either "eff", "aff", "c", "recs", "redAff", "dim", or 'standard'"""
    
    #Run this function if don't already have saved epithelium files to use
    makeSimilar(numRecs, aff_sd, eff_sd, purpose, qspaces, dim)
    
    startTime = time.time()
    
    purp = purpFunction(purpose, aff_sd, eff_sd, numRecs, c, dim)
    
    pdfName = "LigandSat with varying qspaces" + purp
    plotTitle = "Saturation of dPsiBar" + purp
    
    #i = 0
    labelNames = []
    excelNames = []
    end = False
    
    
    # if __name__ == '__main__':
    #     jobs = []
    
    #while i < len(qspaces):
    for i, qspacesItem in enumerate(qspaces):    
        space = []
        #j = 0
        #while j < dim:
        for j in range(dim):    
            space.append((0,qspacesItem))
            #j+=1
        qspace = QSpace(space)


        epith = Epithelium.load("1. SavedEpi_" + str(qspace.size[0]) + purp + ".csv")

        labelNames.append(str(qspace.size[0]) + " qspace")
        excelNames.append("LigandSat with " + str(qspace.size[0]) + " qspace" + purp)
    
        # if i == (len(qspaces) - 1):
        #     end = True

        x = 0
        y = 0
        ID = 0
        odorscenes = []
        #while x < qspace.size[0][1]:
        while x < qunits * 10:
            y = 0
            #while y < qspace.size[1][1]:
            while y < qunits * 10:
                odorscenes.append(Odorscene(x,[Ligand(ID, [x / float(qunits), y / float(qunits)], .004)]))
                y += 1
                ID += 1
            x += 1
        colorMapSumOfSquares(epith, odorscenes, .3, qspace)
      
        # p = multiprocessing.Process(target=dPsiBarSaturation, args=(epith, .01, qspace, pdfName, labelNames[i], excelNames[i], fixed ,c, plotTitle, end, purp, False))
        # jobs.append(p)
        # p.start()
    
    
        #epi, dn, qspace, pdfName, labelName, excelName, fixed eff
        #dPsiBarSaturation(epith, .01, qspace, pdfName, labelNames[i], excelNames[i], fixed ,c, plotTitle, end, purp, graphIt=False)
    
        #i += 1
        
        
        # for j, job in enumerate(jobs):
        #     job.join()



def makeSimilar(numRecs, aff_sd, eff_sd, purpose = "eff", qspaces = [4,10,30], dim = 2):
    """Creates and saves three epithelium determined by qspaces.
    It keeps aff and eff SD identical and only changes means."""
    
    purp = purpFunction(purpose, aff_sd, eff_sd, numRecs, 1, dim)
    
    space = []
    i = 0
    
    # Create a list of tuples ('0', first element of qspaces), list length equals 'dim'
    while i < dim:
        space.append((0, qspaces[0]))
        i += 1
    
    qspace = QSpace(space)
    epith = Epithelium.create(numRecs, dim, qspace, aff_sd, eff_sd) #amt, dim **amt = len(gl) and dim = dim of odorscene
    epith.save("1. SavedEpi_" + str(qspace.size[0]) + purp)
    
    i = 1

    # Iterate over the elements in qspaces (excluding the first one) 
    while i < len(qspaces): 
        space = []
        k = 0
        # Create a new list of tuples using the current element of qspaces
        while k < dim:
            space.append((0,qspaces[i]))
            k += 1
        
        # Create a new Epithelium object
        qspace = QSpace(space)
        epith2 = Epithelium.create(numRecs, dim, qspace, aff_sd, eff_sd)
    
        k = 0

        # Set sdA and sdE attributes of recs in the new Epithelium to match the first Epithelium
        for rec in epith2.recs:
            rec.sdA = epith.recs[k].sdA
            rec.sdE = epith.recs[k].sdE        
            k += 1
    
        # Save the new Epithelium object to a file
        epith2.save("1. SavedEpi_" + str(qspace.size[0]) + purp)
        
        i += 1

def purpFunction(purpose, aff_sd=[0.5, 1.5], eff_sd=[0.05, 1.0], numRecs = 30, c = 1, dim = 2):
    """Returns a string that can be used for titles etc."""
    if purpose == "aff": 
        return ", aff_sd=" + str(aff_sd)
    elif purpose == "eff":
        if len(eff_sd) == 1:
            return ", eff_sd=" + str(eff_sd[0])
        else:
            return ", eff_sd=" + str(eff_sd)
    elif purpose == "c":
        return ", glom_pen=" + str(config.GLOM_PENETRANCE)
    elif purpose == "recs":
        return ", numRecs=" + str(numRecs)
    elif purpose == "redAff":
        num = str(8 + config.PEAK_AFFINITY)
        return ", redAff by 10^" + num
    elif purpose == "dim":
        return ", dim=" + str(dim)
    else: #purpose == "standard"
        return ""
    


def test():

    # runReceptorOdorGraphToolStandAlone()

    testdPsiBarSat(fixed = False, aff_sd = [.5, 1.5], eff_sd = [0.05, 1.0], numRecs = 30, c = 1, dim = 2, qspaces = [4,10,30], purpose = "standard")

    # testdPsiBarSatColorMap(fixed = True, aff_sd = [.5, 1.5], eff_sd = [0.05, 1.0], numRecs = 30, c = 1, dim = 2, qspaces = [4], purpose = "standard", qunits = 4)


    # allGraphsFromExcel(aff_sd = [0.5, 1.5], eff_sd = [0.05, 1.0], numRecs = 30, c = 1, dim = 2, qspaces = [4,10,30], purpose = "standard", rep = 2)

    ####testing varying dimensions
    #testdPsiBarSaturationDim(dims = [2,3,4,5], fixed = False, aff_sd = [.5, 1.5], eff_sd = [.05, 1.0], numRecs = 30, c = 1)
    #dimAllGraphsFromExcel(numRecs = 30, dims = [2,3,4,5], rep = 200)

if __name__ == '__main__':
    test()
