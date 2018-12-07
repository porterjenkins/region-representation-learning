import numpy as np



class GridCell:

    def __init__(self,id,lat_range=None,lon_range=None):
        self.id = id
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.centroid = self.calc_centroid()




    def calc_centroid(self):
        lat_centroid = (self.lat_range[0] + self.lat_range[1])/2.0
        lon_centroid = (self.lon_range[0] + self.lon_range[1])/2.0

        return (lon_centroid, lat_centroid)

