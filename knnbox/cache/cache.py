import torch

class SkipCache:
    def __init__(self, max_size=512, hidden_size=1024):
        self.max_size = max_size
        self.hidden_size = hidden_size

        self.cache_index = 0
        self.cache_keys = torch.zeros([max_size, hidden_size], device='cuda', dtype=torch.float32)
        self.cache_vals = torch.zeros([max_size], device='cuda', dtype=torch.long)


    def _update_cache(self, query):
        query = query.view(-1, 1024)
        update_size = query.size(0)

        bindex = self.cache_index
        eindex = (self.cache_index + update_size) % self.max_size

        if bindex < eindex:
            bindex = self.cache_index
            eindex = self.cache_index + update_size
            self.cache_query[bindex:eindex] = query
        
        else:
            self.cache_query[bindex:] = query[:update_size - eindex]
            self.cache_query[:eindex] = query[update_size - eindex:]

        self.cache_index = eindex
        

    def update_cache(self, query, nmt_indices, knn_indices, retrieve_index):
        nmt_indices = nmt_indices.index_select(0, retrieve_index).squeeze()
        knn_indices = knn_indices.index_select(0, retrieve_index).squeeze()

        should_skip_index = torch.argwhere(nmt_indices == knn_indices).view(-1)

        if should_skip_index.numel() > 0:
            self.update_batch.append(query.index_select(0, should_skip_index))
            self.update_batch_size += should_skip_index.numel()
        
        print(f'knn_cache should skip: {len(should_skip_index)}')
        print(" - - - - - - - - - - ")

        if self.update_batch_size > 256:
            self._update_cache(torch.cat(self.update_batch, dim=0))
            self.update_batch = []
            self.update_batch_size = 0


    def retrieve(self, query):
        distances = torch.cdist(query.view(-1, 1, 1024), self.cache_query.view(1, -1, 1024), p=2)
        min_dists = distances.min(dim=-1).values

        return min_dists