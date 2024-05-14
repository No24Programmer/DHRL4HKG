from torch.utils.data import Dataset


class HDataset(Dataset):
    """

    """
    def __init__(self, triples1):
        self.triples1 = triples1

    def __len__(self):
        return len(self.triples1[0])


    def __getitem__(self, index):
        return self.triples1[0][index], \
               self.triples1[1][index], \
               self.triples1[2][index], \
               self.triples1[3][index], \
               self.triples1[4][index], \
               self.triples1[5][index], \
               self.triples1[6][index], \
               self.triples1[7][index], \
               self.triples1[8][index], \
               self.triples1[9][index], \
               self.triples1[10][index], \
               self.triples1[11][index], \
               self.triples1[12][index]

