import torch


class MeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        query,
        target,
        knn_dist,
        knn_tgt,
        attn,
        left=0,
        right=None,
        move_data_to_gpu=False,
    ):
        self.device = 'cuda:0'
        device = self.device if move_data_to_gpu else query.device

        self.query = query[left:right].to(device)
        self.target = target[left:right].to(device)
        self.knn_dist = knn_dist[left:right].to(device)
        self.knn_tgt = knn_tgt[left:right].to(device)
        self.attn = attn[left:right].to(device)

    def __getitem__(self, index):
        return {
            'query': self.query[index],
            'target': self.target[index],
            'knn_dist': self.knn_dist[index],
            'knn_tgt': self.knn_tgt[index],
            'attn': self.attn[index]
        }

    def __len__(self):
        return len(self.query)
    
    def collate(self, samples):
        def merge(key):
            if len(samples) == 0 or samples[0][key] is None:
                return None
            return torch.stack([sample[key] for sample in samples]).to(self.device)
        
        return {
            'target': merge('target'),
            'net_input': {
                'query': merge('query'),
                'knn_dist': merge('knn_dist'),
                'knn_tgt': merge('knn_tgt'),
                'attn': merge('attn')
            },
        }
