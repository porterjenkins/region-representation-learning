import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid


c = get_config()
grid_size = 50
file = open(c["poi_file"], 'rb')
img_dir = c['path_to_image_dir']
region_grid = RegionGrid(grid_size, poi_file=file, img_dir=img_dir, w_mtx_file=c['flow_mtx_file'],
                         housing_data=c["housing_data_file"], load_imgs=False)

A = region_grid.adj_matrix
D = region_grid.degree_matrix
W = region_grid.weighted_mtx

print(W.shape)
print(A.shape)
print(D.shape)