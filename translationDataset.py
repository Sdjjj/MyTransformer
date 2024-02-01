import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, sentences, word2idx_cn, word2idx_en):
        self.sentences = sentences
        self.word2idx_cn = word2idx_cn
        self.word2idx_en = word2idx_en
    
    def __len__(self):
        return(len(self.sentences))
    
    def __getitem__(self, index):
        # 将句子转换为索引
        sentence_cn = [self.word2idx_cn[word] for word in self.sentences[index][0].split()]
        sentence_en_input = [self.word2idx_en[word] for word in self.sentences[index][1].split()]
        sentence_en_output = [self.word2idx_en[word] for word in self.sentences[index][2].split()]
        return torch.tensor(sentence_cn), torch.tensor(sentence_en_input), torch.tensor(sentence_en_output)
     