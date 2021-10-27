import cv2
import numpy as np
import math


def padding(img, K_size=3):
    H, W, C = img.shape
    pad = K_size // 2
    rows = np.zeros((pad, W, C), dtype=np.uint8)
    cols = np.zeros((H+2*pad, pad, C), dtype=np.uint8)
    img = np.vstack((rows, img, rows))
    img = np.hstack((cols, img, cols))

    return img


def sobel_filter(img, K_size=3):
    H, W, C = img.shape
    pad = K_size // 2
    out = padding(img, K_size=3)
    K_v = np.array([[1., 2., 1.],[0., 0., 0.], [-1., -2., -1.]])
    K_h = np.array([[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]])
    tem = out.copy()
    output = out.copy()
    for h in range(H):
        for w in range(W):
            for c in range(C):
                output[pad+h, pad+w, c] = math.sqrt(np.sum(K_v * tem[h:h+K_size, w:w+K_size, c], dtype=np.float) ** 2 +
                                                    np.sum(K_h * tem[h:h + K_size, w:w + K_size, c], dtype=np.float) ** 2)
    output = np.clip(output, 0, 255)
    output = output[pad:pad + H, pad:pad + W].astype(np.uint8)

    return output

def BGR2GRAY(img):
    H, W, C = img.shape
    out = np.ones((H,W,3))
    for i in range(H):
        for j in range(W):
            out[i,j,:] = 0.299*img[i,j,0] + 0.578*img[i,j,1] + 0.114*img[i,j,2]
    out = out.astype(np.uint8)
    return out


path = 'D:/'
file_in = path + 'Animal.jpg'
file_out = path + 'Sobel_filter.jpg'
img = cv2.imread(file_in)
img = BGR2GRAY(img)
out = sobel_filter(img)
cv2.imwrite(file_out, out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()