from deepwalk.__main__ import process
from config import get_config

class DWparams(object):
    def __init__(self, input, format, undirected, output, number_walks, representation_size, walk_length):
        self.input = input
        self.format = format
        self.undirected = undirected
        self.output = output
        self.number_walks = number_walks
        self.representation_size = representation_size
        self.walk_length = walk_length

        self.max_memory_data_size = 1000000000
        self.seed = 1990
        self.window_size = 5
        self.workers = 1



# Configure project
config = get_config()


args = DWparams(input=config['edge_list_file'],
              format='adjlist',
              undirected=True,
              output=config['deepwalk_file'],
              number_walks=100,
              representation_size=64,
              walk_length=40
              )

process(args)