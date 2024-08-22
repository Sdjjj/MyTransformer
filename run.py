
import torch.nn as nn
from Model import Transformer
from CorpusLoader import CorpusLoader
from torch.utils.data import DataLoader
import torch

from Utilities import read_data
from CorpusLoader import WikiCorpus
from Model import GPT
from Model import GPTTrainer
from Model import TransformerTrainer
import argparse
import nltk
from nltk.data import find

def trainTransformer(): 
    def collate_fn(batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        sentence_cn, sentence_en_in, sentence_en_out = zip(*batch)
        sentence_cn = nn.utils.rnn.pad_sequence(sentence_cn, padding_value=corpus_loader.word2idx_cn['<pad>'],batch_first=True)
        sentence_en_in = nn.utils.rnn.pad_sequence(sentence_en_in, padding_value=corpus_loader.word2idx_en['<pad>'],batch_first=True)
        sentence_en_out = nn.utils.rnn.pad_sequence(sentence_en_out, padding_value=corpus_loader.word2idx_en['<pad>'],batch_first=True)
        return sentence_cn, sentence_en_in, sentence_en_out
    
    # 导入数据集和Transformer模型
    corpus_loader = CorpusLoader(all_sentences_path)
    corpus_loader.process_sentences()
    corpus_loader.build_vocab()
    dataset = corpus_loader.create_dataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    model = Transformer(corpus_loader)
    trainer = TransformerTrainer(model,corpus_loader, dataloader, learning_rate=0.01,epochs=1000)
    trainer.train(checkpoint_path, save_dir)


def trainGPT():
    # 导入数据集和GPT模型
    corpus = WikiCorpus('./GPTDataset/wikitext-103/wiki.train.txt')
    wikiGPT = GPT(corpus)
    trainer = GPTTrainer(wikiGPT, corpus, learning_rate=0.01,epochs=2000)
    trainer.train(save_dir)
    model_save_path = f'./WikiGPT_{trainer.lr}_{trainer.epochs}.pth'
    torch.save(wikiGPT.state_dict(), model_save_path)
    

def testGPT():
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    corpus = WikiCorpus(read_data('./GPTDataset/wikitext-103/wiki.train.txt'))  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_model = GPT(corpus).to(device)
    loaded_model.load_state_dict(torch.load(checkpoint_path))
    
    loaded_model.eval()
    
    input_str = "please tell me"
    greedy_search_output = loaded_model.decode(input_str, strategy='greedy', max_len=25)
    beam_search_output = loaded_model.decode(input_str, strategy='beam_search', 
                                            max_len=25, beam_width=5, reaptition_penalty=1.2)
    print("Input text:", input_str)
    print("Greedy search output:", greedy_search_output)
    print("Beam search output:", beam_search_output)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['trainGPT', 'testGPT', 'trainTransformer'])

    args = parser.parse_args()
    save_dir = './model'
    checkpoint_path = './model/model_epoch_2000_loss_1.47.pth'
    all_sentences_path = './TranslateDataset/all_sentences.txt'
    
    if args.mode == 'trainGPT':
        trainGPT()
    elif args.mode == 'testGPT':
        testGPT()
    elif args.mode == 'trainTransformer':
        trainTransformer()