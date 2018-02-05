import cv2
import numpy as np

lena = cv2.imread('Images/ps0-1-a-2.tiff')
tree = cv2.imread('Images/ps0-1-a-1.tiff')

bt, gt, rt = cv2.split(tree)
bl, gl, rl = cv2.split(lena)

cv2.imwrite("Images/ps0-2-a.tiff", cv2.merge((rt, gt, bt)))

cv2.imwrite("Images/ps0-2-b.tiff", gt)

cv2.imwrite("Images/ps0-2-c.tiff", rt)

mt = bt / 3 + gt / 3 + rt / 3
ml = bl / 3 + gl / 3 + rl / 3

q3 = ml
q3[206:306, 206:306] = mt[206:306, 206:306]

cv2.imwrite("Images/ps0-3.tiff", q3)

print("max: ", np.max(gt))
print("min: ", np.min(gt))
print("mean:", np.mean(gt))
print("std: ", np.std(gt))

q4b = (gt - np.mean(gt)) / np.std(gt) * 10 + np.mean(gt)

cv2.imwrite("Images/ps0-4-b.tiff", q4b)

q4c = gt[1:, 1:]

cv2.imwrite("Images/ps0-4-c.tiff", q4c)

# translate the array from char to int to prevent overflow
gtb = np.array(gt, dtype=int)
q4cb = np.array(q4c, dtype=int)

q4d = gtb[:-1, :-1] - q4cb

# useless as done by opencv
# q4dc = q4d.clip(0, 255)

cv2.imwrite("Images/ps0-4-d.tiff", q4d)

sigma = 25
q5g = gt.astype('int') + np.random.normal(scale=sigma, size=gt.shape)
q5g = q5g.clip(0, 255).astype('uint8')

cv2.imwrite("Images/ps0-5-a.tiff", cv2.merge((bt, q5g, rt)))

q5b = bt.astype('int') + np.random.normal(scale=sigma, size=bt.shape)
q5b = q5b.clip(0, 255).astype('uint8')

cv2.imwrite("Images/ps0-5-b.tiff", cv2.merge((q5b, gt, rt)))


