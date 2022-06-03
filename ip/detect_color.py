import cv2
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator

# import argparse
#
# # Creating argument parser to take image path from command line
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True, help="Image Path")
# args = vars(ap.parse_args())
# img_path = args['image']


# print(img.shape[1] * img.shape[0])
# print(img[0], img[0].shape)
# a, b, c = img[0][0]
# print(a, b, c)


# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
# index = ["color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('../ip/data/colors.csv', names=index, header=None)
# csv = pd.read_csv('data/main_colors.csv', names=index, header=None)

# Reading the image with opencv
img = cv2.imread("/home/selcanyagmuratak/PycharmProjects/Bitirme0/front/static/images/bardak.jpeg")


# function to calculate minimum distance from all colors and get the most matching color
def getColorName(R, G, B, df=csv):
    global cname
    minimum = 10000
    for i in range(len(df)):
        d = abs(R - int(df.loc[i, "R"])) + abs(G - int(df.loc[i, "G"])) + abs(B - int(df.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = df.loc[i, "color_name"]
    return cname


def get_avg_bgr(img):
    B = G = R = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            B += img[y][x][0]
            G += img[y][x][1]
            R += img[y][x][2]

    B = int(B / (img.shape[0] * img.shape[1]))
    G = int(G / (img.shape[0] * img.shape[1]))
    R = int(R / (img.shape[0] * img.shape[1]))

    print(B, G, R)

    return B, G, R


def get_color_histogram(img):
    color_dict = dict()

    B = G = R = 0
    a = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            B += img[y][x][0]
            G += img[y][x][1]
            R += img[y][x][2]

            # color_name = getColorName(R, G, B)
            #
            # if color_name in color_dict:
            #     color_dict[color_name] += 1
            # else:
            #     color_dict[color_name] = 1

            a += 1
            print(a)
    return color_dict


def get_color_dist(img):
    image = img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    image = image.reshape((img.shape[1] * img.shape[0], 3))

    md = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(image)
        o = kmeans.inertia_
        md.append(o)
    print(md)

    kn = KneeLocator(
        range(1, 10),
        md,
        curve='convex',
        direction='decreasing',
        interp_method='polynomial',
    )

    n_cluster = kn.knee
    print(n_cluster)

    main_kmeans = KMeans(n_clusters=n_cluster)
    s = main_kmeans.fit(image)

    labels = main_kmeans.labels_
    print(labels)
    labels = list(labels)

    centroid = main_kmeans.cluster_centers_
    print(centroid)

    percent = []
    color_dict = {}
    for i in range(len(centroid)):
        j = labels.count(i)
        j = j / (len(labels))
        percent.append(j)
        color_dict[i] = j
    print(percent)
    print(color_dict)

    sorted_color_dict = dict(sorted(color_dict.items(), key=lambda item: item[1], reverse=True))
    print(sorted_color_dict)

    color_name_dict = {}
    count = 0

    for keys in sorted_color_dict:
        color_name = getColorName(centroid[int(keys)][0], centroid[int(keys)][1], centroid[int(keys)][2])
        print(color_name)
        color_name_dict[color_name] = sorted_color_dict[keys]
        if count < n_cluster:
            count += 1
        else:
            break

    return color_name_dict


# b, g, r = get_avg_bgr(img)
# color_name = getColorName(r, g, b)
# print(color_name)
#
# return_dict = get_color_histogram(img)
# print(return_dict)

# mydict = get_color_dist(img)
# print(mydict)
