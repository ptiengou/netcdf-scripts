import numpy as np
import great_circle_calculator.great_circle_calculator as gcc

def getGCD(lon, lat):

    GCD = np.zeros(len(lon))
    for i in range(len(lon) - 1):
        GCD[i+1] = gcc.distance_between_points((lon[i], lat[i]), \
                  (lon[i+1], lat[i+1]), unit='kilometers', haversine=True)

    GCD = np.cumsum(GCD)

    return GCD

