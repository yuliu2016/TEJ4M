"""
Implementation of
https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
"""

import math

def merge_pt(pt):
    a, b = zip(*pt)
    return a / len(pt), b / len(pt)

def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def argsort(x):
    return sorted(range(len(x)), key=lambda i: x[i])

class Tracker:
    frames = [([],[])]
    to_keep = 5

    def update(self, new):
        old = self.frames[-1]
        all_ed = []
        sorted_ed = []
        for n in new:
            ed = []
            for o in old:
                ed.append(dist(o, n))
            all_ed.append(ed)
            sorted_ed.append(argsort(ed))

        self.frames.append((new, []))
