import torch
import numpy as np
from knnbox.retriever.utils import retrieve_k_nearest
from concurrent.futures import ThreadPoolExecutor

class LinesCache:
    def __init__(self, datastore_keys, datastore_vals, window_size):

        self.datastore_keys = np.zeros(datastore_keys.shape, dtype=datastore_keys.dtype)
        self.datastore_vals = np.zeros(datastore_vals.shape, dtype=datastore_vals.dtype)
        self.datastore_keys[:] = datastore_keys
        self.datastore_vals[:] = datastore_vals

        self.window_size = window_size
        self.datastore_size = self.datastore_vals.shape[0]

        self.cache_block_num = 4
        self.cache_block_size = 8 * self.window_size
        
        self.cache_keys = None
        self.cache_vals = None
        self.cache_block_index = 0

        self.hits = 0
        self.total = 0

        self.update_num = 0
        self.clear_num = 0
        self.max_update_num = 5
        self.current_update_num = 0

        self.current_position = 0
        self.update_position_cnt = [0] * 100


    def clear(self):
        print(f"update_num: {self.update_num}, clear_num: {self.clear_num}")
        print(f"position_num: {self.update_position_cnt[:20]}")

        self.cache_keys = None
        self.cache_vals = None
        self.cache_block_index = None
        self.clear_num += 1
        self.current_update_num = 0
        self.current_position = 0

    
    def _update_cache(self, indices, bsz_index, window_indices):
        keys_np = self.datastore_keys[window_indices].reshape(-1, self.cache_block_size, 1024)
        vals_np = self.datastore_vals[window_indices].reshape(-1, self.cache_block_size)
        
        # print(f"bsz_index: {bsz_index.shape}")
        # print(f"cache_block_index: {self.cache_block_index.shape}")
        # print(f"cache_block_index[bsz_index]: {self.cache_block_index[bsz_index].shape}")

        update_index = bsz_index * self.cache_block_num + self.cache_block_index[bsz_index]

        self.cache_keys.view(-1, self.cache_block_size, 1024)[update_index] \
            = torch.tensor(keys_np, device='cuda', dtype=torch.float32)
        
        self.cache_vals.view(-1, self.cache_block_size)[update_index] \
            = torch.tensor(vals_np, device='cuda', dtype=torch.long)

        self.cache_block_index[bsz_index] += 1
        self.cache_block_index %= self.cache_block_num
        

    def init_cache(self, bsz):
        self.cache_keys = torch.zeros(
            [bsz, self.cache_block_num, self.cache_block_size, 1024], device='cuda', dtype=torch.float32)
        self.cache_vals = torch.zeros(
            [bsz, self.cache_block_num, self.cache_block_size], device='cuda', dtype=torch.long)
        self.cache_block_index = torch.zeros([bsz], device='cuda', dtype=torch.long)
    

    # @profile
    def update(self, indices, bsz_index):

        if self.current_update_num == self.max_update_num:
            return

        window_indices = torch.cat([indices + i for i in range(self.window_size)], dim=-1).view(-1)
        window_indices.clamp_(max=(self.datastore_size - 1))
        self._update_cache(indices, bsz_index, window_indices)
        
        self.update_num += 1
        self.current_update_num += 1
        self.current_position += 1

    
    def empty(self):
        return self.cache_keys is None
    

    def index_select(self, index):
        self.cache_keys = torch.index_select(self.cache_keys, dim=0, index=index)
        self.cache_vals = torch.index_select(self.cache_vals, dim=0, index=index)


    def update_bsz(self, index, bsz, new_bsz):
        self.cache_keys = self.cache_keys.view(bsz, -1)[index].view(new_bsz, self.cache_block_num, -1, 1024)
        self.cache_vals = self.cache_vals.view(bsz, -1)[index].view(new_bsz, self.cache_block_num, -1)


    # @profile
    def retrieve(self, query, k, retrieve_index=None):
        bsz = self.cache_vals.size(0)
        # print(f"query: {query.shape}")
        # print(f"cache_keys: {self.cache_keys.shape}")
        # print(f"cache_vals: {self.cache_vals.shape}")

        if retrieve_index is not None:
            # print(f"retrieve_index: {retrieve_index.shape}")
            # print(f"retrieve_index: {retrieve_index}")

            cache_keys = self.cache_keys.view(bsz, -1, 1024).index_select(0, retrieve_index)
            cache_vals = self.cache_vals.view(bsz, -1).index_select(0, retrieve_index)

            distances = torch.sum((cache_keys - query.unsqueeze(1)) ** 2, dim=-1)
            knn_dist, indices = distances.topk(k=k, dim=-1, largest=False)
            knn_tgt = torch.gather(cache_vals, -1, indices)
            return knn_dist, knn_tgt
        else:
            distances = torch.sum((self.cache_keys.view(bsz, -1, 1024) - query) ** 2, dim=-1)
            knn_dist, indices = distances.topk(k=k, dim=-1, largest=False)
            # print(f"knn_dist: {knn_dist.shape}")
            # print(f"indices: {indices.shape}")
            
            knn_tgt = torch.gather(self.cache_vals.view(bsz, -1), -1, indices)
            return knn_dist.unsqueeze(1), knn_tgt.unsqueeze(1)


class LinesRetriever:
    def __init__(self, datastore, k, window_size=4):
        self.datastore = datastore
        self.k = k
        self.window_size = window_size
        
        self.cache = LinesCache(
            datastore_keys=datastore.datas["keys"].data,
            datastore_vals=datastore.datas["vals"].data,
            window_size=window_size,
        )
        self.results = None

    
    # @profile
    def retrieve(self, query, return_list = ["keys", "vals", "distances"], bsz=None, distance_threshold=50.0, retrieve_index=None):
        # load the faiss index if haven't loaded
        ret = {}
        query = query.detach()

        # print(f"query: {query.shape}")

        if self.cache.empty():
            use_datastore_index = torch.arange(start=0, end=query.size(0), device=query.device)
            query_using_datastore = query
            hit_percent = 0
            self.cache.init_cache(bsz if bsz is not None else query.size(0))

        else:
            if retrieve_index is not None:
                knn_dist, knn_tgt = self.cache.retrieve(query, self.k, retrieve_index)
            else:
                knn_dist, knn_tgt = self.cache.retrieve(query, self.k)

            min_knn_dist = knn_dist.min(dim=-1).values.squeeze()
            use_datastore_index = (min_knn_dist > distance_threshold).nonzero().squeeze()

            # print(f"min_knn_dist: {min_knn_dist}")
            # print(f"min_knn_dist: {min_knn_dist.shape}")
            # print(f"use_datastore_index: {use_datastore_index}")
            
            hit_percent = 1 - use_datastore_index.numel() / query.size(0)

            self.cache.update_position_cnt[self.cache.current_position] += query.size(0) - use_datastore_index.numel()

            if hit_percent > 0:
                ret["distances"] = knn_dist
                ret["vals"] = knn_tgt
                query_using_datastore = query[use_datastore_index]

                self.cache.total += query.size(0)
                self.cache.hits += query.size(0) - use_datastore_index.numel()

            else:
                query_using_datastore = query
        
        query = query_using_datastore
        faiss_results = retrieve_k_nearest(query, self.datastore.faiss_index["keys"], self.k)
        indices = faiss_results["indices"].cpu().numpy()

        if "distances" in ret:
            ret["distances"][use_datastore_index] = faiss_results["distances"]
            ret["vals"][use_datastore_index] = torch.tensor(self.datastore["vals"].data[indices], device=query.device)
        else:
            ret["distances"] = faiss_results["distances"]
            ret["vals"] = torch.tensor(self.datastore["vals"].data[indices], device=query.device)

        # update cache
        if self.cache.empty() or hit_percent < 0.1:
            if retrieve_index is not None:
                use_datastore_index = retrieve_index[use_datastore_index]
            self.cache.update(faiss_results["indices"].cpu(), bsz_index=use_datastore_index)

        # print(f"distances: {ret['distances'].shape}")
        # print(f"vals: {ret['vals'].shape}")

        self.results = ret
        return ret