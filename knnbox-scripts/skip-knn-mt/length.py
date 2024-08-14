from knnbox.common_utils import read_config, Memmap
from knnbox.datastore import Datastore
import sys

domain = sys.argv[1]

valid_datastore = Datastore(path=f'../../datastore/vanilla/{domain}/valid')
valid_config = read_config(f'../../datastore/vanilla/{domain}/valid')["data_infos"]
positions_tgt = Memmap(f'../../datastore/vanilla/{domain}/valid/src_lengths.npy', shape=valid_config["src_lengths"]["shape"],
                          dtype=valid_config["src_lengths"]["dtype"], mode="r+")
print(sum(positions_tgt.data[:])/len(positions_tgt.data[:]))
