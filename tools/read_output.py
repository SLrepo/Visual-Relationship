import pickle

f = open("out", "rb")
res1 = pickle.load(f)
print(res1)
f.close()