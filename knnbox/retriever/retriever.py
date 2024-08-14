import torch
from knnbox.retriever.utils import retrieve_k_nearest

class Retriever:
    def __init__(self, datastore, k):
        self.datastore = datastore
        self.k = k
        self.results = None


    def retrieve(self, query, return_list = ["vals", "distances"], k = None, remove_first = False, bsz = None, retrieve_index = None, keys_name = "keys"):
        r""" 
        retrieve the datastore, save and return results 
        if parameter k is provided, it will suppress self.k
        """

        k = k if k is not None else self.k

        # load the faiss index if haven't loaded
        if keys_name not in return_list:
            if not hasattr(self.datastore, "faiss_index") or \
                        self.datastore.faiss_index is None or keys_name not in self.datastore.faiss_index:
                self.datastore.load_faiss_index(keys_name, move_to_gpu=True)

        query = query.detach()
        faiss_results = retrieve_k_nearest(query, self.datastore.faiss_index[keys_name], k if not remove_first else k + 1)

        ret = {}
        if "distances" in return_list:
            ret["distances"] = faiss_results["distances"] if not remove_first else faiss_results["distances"][...,1:]
        if "indices" in return_list:
            ret["indices"] = faiss_results["indices"] if not remove_first else faiss_results["indices"][...,1:]
        if "k" in return_list:
            ret["k"] = k
        if "query" in return_list:
            ret["query"] = query

        # other information get from self.datastores.datas using indices, for example `keys` and `vals`
        indices = faiss_results["indices"] if not remove_first else faiss_results["indices"][...,1:]
        indices = indices.cpu().numpy()
        for data_name in return_list:
            if data_name not in ["distances", "indices", "k", "query"]:
                assert data_name in self.datastore.datas, \
                                    "You must load the {} of datastore first".format(data_name)
                ret[data_name] = torch.tensor(self.datastore[data_name].data[indices], device=query.device)
        
        self.results = ret # save the retrieved results
        return ret
    
        