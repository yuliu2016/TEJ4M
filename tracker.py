"""
Implementation of
https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

Requires Python 3.7+ for guaranteed ordered dicts

Algo:

Find centroids of objects
Compare the new count and old count of centroids to determine whether object entered or left the image frame
Compute the Euclidean distance between the new centroids and the old centroids. The pair of closest old/new centroid
is determined to be the motion of the object.
This method is based on the assumption that the closest change is the motion. If multiple objects move at different
speeds and moved for the same amount of time, it would not work
A more complex method is to detect features of the object to see where they go
This is an easy way to make a direction-sensitive detector, since the distance can be vectors
"""

import math

def merge_point(points):
    return tuple(sum(x) / len(x) for x in zip(*points))

def dist(a, b):
    return math.hypot(*(b[i] - a[i] for i in range(min(len(a), len(b)))))

def argsort(x, key=None, reverse=False):
    if key is None:
        key = lambda i: x[i]
    else:
        key = lambda i: key(x[i])
    return sorted(range(len(x)), key=key, reverse=reverse)

class Tracker:
    objects = {}
    def update(self, new_locations):
        vector_dist = []
        old_locations = self.objects.values()
        for i, new_location in enumerate(new_locations):
            ed = []
            for j, old_location  in enumerate(old_locations):
                ed.append((i, j, dist(old_location, new_location)))
            ed.sort(key=lambda x: x[2])
            vector_dist.append(ed)
        vector_dist.sort(key=lambda x: x[0][2])
