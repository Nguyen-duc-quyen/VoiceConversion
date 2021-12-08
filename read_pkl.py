import pickle

#with open('results.pkl', 'rb') as f:
with open('metadata.pkl', 'rb') as f: 
    data = pickle.load(f)

for obj in data:
    print(len(obj))
    name = obj[0]
    c = obj[1]
    d = obj[2]
    print(c)
