import numpy as np
import registration
from utils import *
import cv2

# designate image path here
IX_path = '../image/enhance.jpg'
IY_path = '../image/label.jpg'

IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)

reg = registration.CNN()
X, Y, Z = reg.register(IX, IY)

# generate regsitered image using TPS
registered = tps_warp(Y, Z, IY, IX.shape)

res = np.zeros(shape=(IX.shape[0], IX.shape[1] * 2, 3), dtype=np.uint8)
res[:, :IX.shape[1], :] = IX
res[:, IX.shape[1]:, :] = registered
print("The number of matching points: %d" % len(X))
for i, pnt in enumerate(X):
    src_x = int(pnt[1])
    src_y = int(pnt[0])
    dst_x = int(Z[i][1] + IX.shape[1])
    dst_y = int(Z[i][0])

    cv2.line(res, (src_x, src_y), (dst_x, dst_y), (255, 0, 0), 2)
    cv2.circle(res, (src_x, src_y), 5, (0, 255, 0), -1)
    cv2.circle(res, (dst_x, dst_y), 5, (0, 255, 0), -1)

cv2.imwrite('res.jpg', res)
