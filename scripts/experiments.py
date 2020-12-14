# Nathaniel Gordon, CS4100 Final Project
import json
import numpy as np
import collections
from scipy import optimize
import random
from multiprocessing import Pool

# Evaluation function
def score(comb, db, pool):
    selected = []
    for c in comb:
        selected.append(int(c))

    # Return 0 if duplicate cards
    if len(selected) != len(set(selected)):
        return 0

    # Count how many cards break the curve
    overCurve = 0
    curveTally = np.zeros(7)
    idealCurve = [45, 2, 7, 6, 4, 3, 1]
    for selind in selected:
        cmc = db[pool[selind]]['convertedManaCost']
        if cmc < 6:
            if curveTally[cmc] == idealCurve[cmc]:
                overCurve = overCurve + 1
            else:
                curveTally[cmc] = curveTally[cmc] + 1
        else:
            if curveTally[6] == idealCurve[6]:
                overCurve = overCurve + 1
            else:
                curveTally[6] = curveTally[6] + 1
    curveScore = (len(comb) - overCurve) / len(comb)

    # Sum the total value of all cards in the deck
    cardScore = 0
    for selind in selected:
        cardScore = cardScore + db[pool[selind]]['value']
    
    return -1*(curveScore*cardScore)

# Similar to score(), but returns more feedback
def getDeckStats(deck, db, pool):
    # Count how many cards break the curve
    overCurve = 0
    curveTally = np.zeros(7)
    idealCurve = [45, 2, 7, 6, 4, 3, 1]
    for card in deck:
        cmc = db[card]['convertedManaCost']
        if cmc < 6:
            if curveTally[cmc] == idealCurve[cmc]:
                overCurve = overCurve + 1
            else:
                curveTally[cmc] = curveTally[cmc] + 1
        else:
            if curveTally[6] == idealCurve[6]:
                overCurve = overCurve + 1
            else:
                curveTally[6] = curveTally[6] + 1
    curveScore = float(len(deck) - overCurve) / float(len(deck))

    # Sum the total value of all cards in the deck
    cardScore = 0
    for card in deck:
        cardScore = cardScore + db[card]['value']
    
    return (cardScore, curveScore)

# Uniqueness constraint for optimizers
def uniqueConstraint(x):
    y=[]
    for xi in x:
        y.append(int(xi))
    if len(y) != len(set(y)):
        return 0
    return 1

# Perform an experimental trial
def runExp(key):
    key = str(key)
    print('Running trial', key)

    pool = pools[key]
    expDict = {'Initial': {}, 'Annealing': {}, 'Genetic': {}, 'Random': {}}

    expDict['Initial']['Deck']=[]
    for card in pool:
        expDict['Initial']['Deck'].append(card)

    expDict['Initial']['Score']=0

    # Remove basic lands
    tbr = []
    for card in pool:
        if 'Basic Land' in db[card]['type']:
            tbr.append(card)
    for card in tbr:
        pool.remove(card)

    # Remove underrepresented colors
    colorMembers = []
    for card in pool:
        for cardColor in db[card]['colorIdentity']:
            colorMembers.append(cardColor)
    
    colorRef = ['R', 'W', 'G', 'B', 'U']
    colorCounts = [colorMembers.count(c) for c in colorRef]
    removeColors = []
    for i in range(3):
        minColorInd = np.argmin(colorCounts)
        removeColors.append(colorRef[minColorInd])
        del colorRef[minColorInd]
        del colorCounts[minColorInd]

    tbr = []
    for card in pool:
        if len(list(set(removeColors).intersection(db[card]['colorIdentity']))) > 0:
            tbr.append(card)
    for card in tbr:
        pool.remove(card)

    varbound=np.array([[0,len(pool)-1]]*23)

    nlc = optimize.NonlinearConstraint(uniqueConstraint, .5, 1.5)

    annealRes = optimize.dual_annealing(score, args=(db, pool), bounds=varbound, maxiter=10000)
    if annealRes.fun == 0:
        print("Annealing Score:", annealRes.fun)
        return None

    genRes = optimize.differential_evolution(score, varbound, args=(db, pool), constraints=(nlc))
    randResSel = random.sample(range(len(pool)), 23)

    print("Annealing Score:", annealRes.fun)
    print("Genetic Score:", genRes.fun)

    if annealRes.fun < 0 and genRes.fun < 0:

        expDict['Annealing']['Deck']=[]
        for x in annealRes.x:
            expDict['Annealing']['Deck'].append(db[pool[int(x)]]['name'])

        expDict['Annealing']['Score']=annealRes.fun

        expDict['Annealing']['Card Score'], expDict['Annealing']['Curve Score'] = getDeckStats(expDict['Annealing']['Deck'], db, pool)

        expDict['Genetic']['Deck']=[]
        for x in genRes.x:
            expDict['Genetic']['Deck'].append(db[pool[int(x)]]['name'])

        expDict['Genetic']['Score']=genRes.fun

        expDict['Genetic']['Card Score'], expDict['Genetic']['Curve Score'] = getDeckStats(expDict['Genetic']['Deck'], db, pool)

        expDict['Random']['Deck']=[]
        for x in randResSel:
            expDict['Random']['Deck'].append(db[pool[int(x)]]['name'])

        expDict['Random']['Card Score'], expDict['Random']['Curve Score'] = getDeckStats(expDict['Random']['Deck'], db, pool)

        expDict['Random']['Score']= - expDict['Random']['Card Score']*expDict['Random']['Curve Score']
    
        return expDict

if __name__ == "__main__":

    # Load card database
    with open("data/m19cards.json", "r") as read_file:
        db = json.load(read_file)

    # Load trial pools
    with open("data/trialPools.json", "r") as read_file:
        pools = json.load(read_file)

    # Run experiment trials with pool
    trials = 2000
    resDict = {}

    mp = Pool()
    results = mp.map(runExp, range(trials))

    # Convert results to serializable JSON dict
    for i in range(len(results)):
        if results[i]:
            resDict[str(i)] = results[i]

    # Serializing json  
    json_object = json.dumps(resDict, indent = 4) 
  
    # Writing to sample.json 
    with open("algOut.json", "w") as outfile: 
        outfile.write(json_object) 
