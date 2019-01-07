import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans

f = open('trees.png', 'rb')
bitimg = []
img = image.open(f)
img = img.convert('RGB')
m, n = img.size
for i in range(m):
    for j in range(n):
        x, y, z = img.getpixel((i, j))
        bitimg.append([x / 256.0, y / 256.0, z / 256.0])
f.close()
bitimg = np.mat(bitimg)

label = KMeans(n_clusters=20).fit_predict(bitimg)
label = label.reshape([m,n])
newimg = image.new("L", (m, n))

for i in range(m):
    for j in range(n):
        newimg.putpixel((i,j), int(256/(label[i][j]+1)))

newimg.save("newtrees.jpg", "JPEG")
