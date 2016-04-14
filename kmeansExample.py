import sys
import math
import random
import subprocess

PLOTLY_USERNAME = None
PLOTLY_KEY = None

if PLOTLY_USERNAME:
    from plotly import plotly


def genPoints():
    with open("spambase.csv") as f:
        for line in f:
            arr = []
            arr = line.split(",")
            #print "size of line::",len(arr)
            #arr = arr[:-1]
            #print "size of line after -1::",len(arr) 
            #raw_input()
            farr = []
            for s in arr:
                farr.append(float(s)) 
            yield Point(farr)

def main2():
    print("\n\n\n\nKMeans::")
    num_points = 10
    dimensions = 57
    lower = 0
    upper = 200
    num_clusters = 2
    opt_cutoff = sys.float_info.min
    #points = [makeRandomPoint(dimensions, lower, upper) for i in xrange(num_points)]
    points = [p for p in genPoints()]
    print "no of points::", len(points)
    tr_size = len(points)
    clusters = kmeans(points, num_clusters, opt_cutoff)
    if len(clusters[0].points) > len(clusters[1].points):
        print "no of spam =", len(clusters[1].points), "\nno of ham =", len(clusters[0].points)
        ch_c,cs_c=0,0
        for p in clusters[0].points:
            if p.coords[-1] == 0:
              ch_c = ch_c + 1
        for p in clusters[1].points:
            if p.coords[-1] == 1:
              cs_c = cs_c + 1
        print "no of spam correctly classified =", cs_c, "\nno of spam wrongly classified =", len(clusters[1].points) - cs_c
        print "\nno of ham correctly classified =", ch_c, "\nno of ham wrongly classified =", len(clusters[0].points) - ch_c
    else:
        print "no of spam =", len(clusters[0].points), "\nno of ham =", len(clusters[1].points)
        ch_c,cs_c=0,0
        for p in clusters[1].points:
            if p.coords[-1] == 0:
              ch_c = ch_c + 1
        for p in clusters[0].points:
            if p.coords[-1] == 1:
              cs_c = cs_c + 1
        print "no of spam correctly classified(TN) =", cs_c, "\nno of spam wrongly classified(FN) =", len(clusters[0].points) - cs_c
        print "\nno of ham correctly classified(TP) =", ch_c, "\nno of ham wrongly classified(FP) =", len(clusters[1].points) - ch_c
        print "accuracy = ", (float(ch_c+cs_c)*100)/tr_size, "%"   
    

class Point:
    def __init__(self, coords):
        self.coords = coords
        self.n = len(coords)-1
        
    def __repr__(self):
        return str(self.coords)

class Cluster:
    def __init__(self, points):
        if len(points) == 0: raise Exception("ILLEGAL: empty cluster")
        self.points = points
        self.n = points[0].n
        for p in points:
            if p.n != self.n: raise Exception("ILLEGAL: wrong dimensions")
        self.centroid = self.calculateCentroid()
        
    def __repr__(self):
        return str(self.points)
    
    def update(self, points):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid) 
        return shift
    
    def calculateCentroid(self):
        numPoints = len(self.points)
        coords = [p.coords for p in self.points]
        unzipped = zip(*coords)
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
        
        return Point(centroid_coords)

def kmeans(points, k, cutoff):    
    initial = random.sample(points, k)
    clusters = [Cluster([p]) for p in initial]
    loopCounter = 0
    while True:
        lists = [ [] for c in clusters]
        clusterCount = len(clusters)
        loopCounter += 1
        for p in points:
            smallest_distance = getDistance(p, clusters[0].centroid)
            clusterIndex = 0
            for i in range(clusterCount - 1):
                distance = getDistance(p, clusters[i+1].centroid)
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            lists[clusterIndex].append(p)
        
        biggest_shift = 0.0
        for i in range(clusterCount):
            shift = clusters[i].update(lists[i])
            biggest_shift = max(biggest_shift, shift)
        
        if biggest_shift < cutoff:
            print "Converged after %s iterations" % loopCounter
            break
    return clusters

def getDistance(a, b):
    if a.n != b.n:
        raise Exception("ILLEGAL: non comparable points")
    
    ret = reduce(lambda x,y: x + pow((a.coords[y]-b.coords[y]), 2),range(a.n),0.0)
    return math.sqrt(ret)

def makeRandomPoint(n, lower, upper):
    p = Point([random.uniform(lower, upper) for i in range(n)])
    return p
