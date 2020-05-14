import pickle

with open("result.pickle","wb") as f:
    pickle.dump([],f, pickle.HIGHEST_PROTOCOL)