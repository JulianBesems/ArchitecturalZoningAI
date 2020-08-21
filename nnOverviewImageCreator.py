import csv
import numpy as np
from PIL import Image

with open("nnWeights/population5c.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    imageArray = []
    for r in reader:
        imageArray.append(r)

im2 = Image.new('RGB', [len(imageArray), int(len(imageArray[0])/8)], 255)
data = im2.load()

for index in range(8):
    #imageList = []
    for i in range(len(imageArray)):
        #row = []
        for j in range(index * int(len(imageArray[0])/8), (index + 1) * int(len(imageArray[0])/8)):
            v = float(imageArray[i][j])
            if v < 0:
                c = int(abs(v * 255))
                data[i,j- index * int(len(imageArray[0])/8)] = (255, 255-c, 255-c)
                # If there is a negative (impeding) weight, make the line red
            elif v > 0:
                c = int(abs(v * 255))
                data[i,j- index * int(len(imageArray[0])/8)] = (255-c, 255-c, 255)
            elif abs(v) > 0.05:
                data[i,j- index * int(len(imageArray[0])/8)] = (255, 255, 255)
            else:
                c = int(abs(v * 225))
                data[i,j- index * int(len(imageArray[0])/8)] = (c, c, c)

            #row.append(colour)
        #imageList.append(row)

    """for row in imageArray:
        for cell in row:
            r = int(max(float(cell)*255, 0))
            g = 0
            b = int(max(-1*float(cell)*255, 0))
            cell = [r,g,b]

    imageArrayNP = np.array(imageList)
    im = Image.fromarray(imageArrayNP, "RGB")
    im.save("population6_" + str(index) + ".jpeg", "JPEG")"""

    im2.save("population5c_" + str(index) + ".jpeg", "JPEG")
