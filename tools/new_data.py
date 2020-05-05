import numpy as np
import pickle


instanceA = [246, 295, 290, 342, 1, 56]
instanceB = [49, 261, 484, 630, 1, 6]
obj = [instanceA, instanceB]
res = [np.array(obj)]
instanceA = [274, 196, 335, 251, 1, 56]
instanceB = [71, 29, 422, 384, 1, 55]
obj = [instanceA, instanceB]
res.append(np.array(obj))

outfile = open("dets",'wb')
pickle.dump(res, outfile)