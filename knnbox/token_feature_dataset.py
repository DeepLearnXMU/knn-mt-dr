import torch
import pickle

class TokenFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, fert_data, freq_data, knn_prob, nmt_prob, nmt_max, nmt_entropy, ctext, ngram=0):
        super().__init__()

        self.freq_data = freq_data
        self.fert_data = fert_data
        self.knn_prob = knn_prob
        self.nmt_prob = nmt_prob
        self.nmt_max = nmt_max
        self.nmt_entropy = nmt_entropy
        self.ctext = ctext

        if ngram == 0 and self.freq_data != None:
            self.ngram = freq_data[0].numel()
        else:
            self.ngram = ngram

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, index):
        
        return{
            "id": torch.tensor(index).long(),
            "ctxt": self.ctext[index],
            "nmt_entropy": self.nmt_entropy[index],
            "nmt_max": self.nmt_max[index],
            "freq": self.freq_data[index],
            "fert": self.fert_data[index],
            "nmt_prob": self.nmt_prob[index],
            "knn_prob": self.knn_prob[index],
        }

    def __len__(self):
        return len(self.ctext)

    def collater(self, samples):
        def merge(key, dtype=torch.float32):
            if len(samples) == 0 or samples[0][key] is None:
                return None
            return torch.stack([s[key] for s in samples]).to(self.device)

        batch = {
            'feature':{
                'ctxt': merge('ctxt'),
                'nmt_entropy': merge('nmt_entropy').unsqueeze(1),
                'nmt_max': merge('nmt_max').unsqueeze(1),
                'freq': merge('freq').unsqueeze(1),
                'fert': merge('fert').unsqueeze(1),
            },
            'id': merge('id'),
            'nmt_prob': merge('nmt_prob'),
            'knn_prob': merge('knn_prob'),
        }
        return batch

    def get_nfeature(self, feature_name):
        return self.__getitem__(0)[feature_name].numel()