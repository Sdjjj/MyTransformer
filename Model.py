import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from nltk.tokenize import word_tokenize
from tqdm import tqdm


d_k = 64
d_v = 64
d_embedding = 512
n_heads = 8
batch_size = 1
n_encoder_layers = 6
n_decoder_layers = 6

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 正弦位置编码表函数
def sine_pos_enc_table(n_position, dim):
    angle_table = np.zeros((n_position, dim))
    for i in range(n_position):
        for j in range(dim):
            angle = i / np.power(1000, 2*j/dim)
            angle_table[i][j] = angle

    angle_table[:, 0::2] = np.sin(angle_table[:, 0::2])
    angle_table[:, 1::2] = np.cos(angle_table[:, 1::2])
    return torch.FloatTensor(angle_table)


# 生成填充注意力掩码函数
def padding_atten_mask(seq_q: torch.Tensor, seq_k: torch.Tensor):
    batch_size_k, len_q = seq_q.size()
    batch_size_k, len_k = seq_k.size()

    pad_mask = seq_k.data.eq(0).unsqueeze(1)
    pad_mask = pad_mask.expand(batch_size_k, len_q, len_k)

    return pad_mask


# 生成后续注意力掩码函数
def subsequent_atten_mask(seq: torch.Tensor):
    atten_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(atten_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


# GPT集束解码策略
def beam_search(model, input_str, max_len=5, beam_width=5, reaptition_penalty=1.2):
    model.eval()
    input_str = word_tokenize(input_str)
    input_tokens = [model.corpus.vocab[token] for token in input_str if token in model.corpus.vocab]
    if len(input_tokens) == 0:
        return         
    candidates = [(input_tokens, 0.0)]    
    final_results = []

    with torch.no_grad():

        for _ in range(max_len):
            new_candidates = []            
            for candidate, candidate_score in candidates:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = torch.LongTensor(candidate).unsqueeze(0).to(device)
                outputs = model(inputs)                
                logits = outputs[:, -1, :]
                for token in set(candidate):
                    logits[0, token] /= reaptition_penalty
                logits[0, model.corpus.vocab["<pad>"]] = -1e9                
                scores, next_tokens = torch.topk(logits, beam_width, dim=-1)
                for score, next_token in zip(scores.squeeze(), next_tokens.squeeze()):
                    new_candidate = candidate + [next_token.item()]                    
                    new_score = candidate_score - score.item()                    
                    if next_token.item() == model.corpus.vocab["<eos>"]:
                        final_results.append((new_candidate, new_score))
                    else:
                        new_candidates.append((new_candidate, new_score))
            candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    if final_results:
        best_candidate, _ = sorted(final_results, key=lambda x: x[1])[0]
    else:
        best_candidate, _ = sorted(candidates, key=lambda x: x[1])[0]
    output_str = " ".join([model.corpus.idx2word[token] for token in best_candidate])
    return output_str


# GPT贪心解码策略
def greedy_search(model, input_str, max_len=25):
    model.eval()
    input_str = word_tokenize(input_str)
    input_tokens = [model.corpus.vocab[token] for token in input_str if token in model.corpus.vocab]
    if len(input_tokens) == 0:
        return "No valid tokens in input."      
    output_tokens = input_tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eos_token = model.corpus.vocab["<eos>"]
    with torch.no_grad():
        for _ in range(max_len):
            inputs = torch.LongTensor(output_tokens).unsqueeze(0).to(device)
            outputs = model(inputs)                
            logits = outputs[:, -1, :]
            _, next_token = torch.topk(logits, 1, dim=-1)
            if next_token.item() == model.corpus.vocab["<eos>"]:
                break
            
            output_tokens.append(next_token.item())
    output_str = " ".join([model.corpus.idx2word[token] for token in output_tokens])
    return output_str
    

# 定义缩放点积注意力类
class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, atten_mask: torch.Tensor):
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        atten_mask = atten_mask.bool()
        scores.masked_fill_(atten_mask, -1e9)
        # weights = nn.Softmax(scores, dim=1)
        weights = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(weights, V)
        return context, weights


# 定义多头注意力类
class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, n_heads*d_k)
        self.W_K = nn.Linear(d_embedding, n_heads*d_k)
        self.W_V = nn.Linear(d_embedding, n_heads*d_v)
        self.linear = nn.Linear(n_heads*d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_Q(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_Q(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context: torch.Tensor
        weights: torch.Tensor
        context, weights = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads*d_v)

        output = self.linear(context)
        output = self.layer_norm(residual+output)

        return output, weights


# 定义逐位置前向传播网络
class PosByPosFeedForwardNet(nn.Module):
    def __init__(self) -> None:
        super(PosByPosFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_embedding,
                               out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=2048, out_channels=d_embedding, kernel_size=1)
        self.layer_normal = nn.LayerNorm(d_embedding)

    def forward(self, inputs: torch.Tensor):
        residual = inputs

        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        
        output = self.layer_normal(output + residual)

        return output


# 定义编码器层类
class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super(EncoderLayer, self).__init__()
        self.multi_head_atten = MultiHeadAttention()
        self.feed_forward = PosByPosFeedForwardNet()

    def forward(self, encode_input: torch, atten_mask):
        enc_outputs, enc_weights = self.multi_head_atten(encode_input, encode_input, encode_input, atten_mask)
        enc_outputs = self.feed_forward(enc_outputs)
        return enc_outputs, enc_weights


# 定义编码器类
class Encoder(nn.Module):
    def __init__(self, corpus) -> None:
        super(Encoder, self).__init__()
        self.src_embeding = nn.Embedding(corpus.src_vocab, d_embedding)
        self.pos_embeding = nn.Embedding.from_pretrained(sine_pos_enc_table(corpus.src_len+1, d_embedding), freeze=True)
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(n_encoder_layers))

    def forward(self, enc_inputs: torch.Tensor):
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1).unsqueeze(0).to(enc_inputs)
        enc_outputs = self.src_embeding(enc_inputs) + self.pos_embeding(pos_indices)
        enc_attn_mask = padding_atten_mask(enc_inputs, enc_inputs)
        enc_self_attn_weights = []

        for layer in self.layers:
            enc_outputs, weights = layer(enc_outputs, enc_attn_mask)
            enc_self_attn_weights.append(weights)

        return enc_outputs, enc_self_attn_weights


# 定义解码器层类
class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super(DecoderLayer, self).__init__()
        self.dec_self_atten = MultiHeadAttention()
        self.dec_enc_atten = MultiHeadAttention()
        self.pos_feed_forward = PosByPosFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_atten_mask, dec_enc_atten_mask):
        dec_outputs, dec_weights = self.dec_self_atten(dec_inputs, dec_inputs, dec_inputs, dec_self_atten_mask)
        dec_outputs, dec_enc_weights = self.dec_enc_atten(dec_outputs, enc_outputs, enc_outputs, dec_enc_atten_mask)
        dec_outputs = self.pos_feed_forward(dec_outputs)
        return dec_outputs, dec_weights, dec_enc_weights

    
# 定义解码器类
class Decoder(nn.Module):
    def __init__(self, corpus) -> None:
        super(Decoder, self).__init__()
        self.tgt_embeding = nn.Embedding(corpus.tgt_vocab, d_embedding)
        self.pos_embeding = nn.Embedding.from_pretrained(sine_pos_enc_table(corpus.tgt_len+1, d_embedding), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_decoder_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        pos_indices = torch.arange(1, dec_inputs.size(1)+1).unsqueeze(0).to(dec_inputs)
        dec_outputs = self.tgt_embeding(dec_inputs) + self.pos_embeding(pos_indices)

        dec_pad_mask = padding_atten_mask(dec_inputs, dec_inputs)
        dec_subsquent_mask = subsequent_atten_mask(dec_inputs)
        
        dec_self_mask = torch.gt((dec_pad_mask.to(device) + dec_subsquent_mask.to(device)), 0)

        dec_enc_mask = padding_atten_mask(dec_inputs, enc_inputs)

        dec_self_weights, dec_enc_weights = [], []
        for layer in self.layers:
            dec_outputs, dec_self_weight, dec_enc_weight = layer(dec_outputs, enc_outputs, dec_self_mask, dec_enc_mask)
            dec_self_weights.append(dec_self_weight)
            dec_enc_weights.append(dec_enc_weight)

        return dec_outputs, dec_self_weights, dec_enc_weights
   
    
# 定义Transformerm模型
class Transformer(nn.Module):
    def __init__(self, corpus) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder(corpus)
        self.decoder = Decoder(corpus)
        self.projection = nn.Linear(d_embedding, corpus.tgt_vocab, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_weights = self.encoder(enc_inputs)
        dec_outputs, dec_self_weights, dec_enc_weights = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        dec_logits = self.projection(dec_outputs)

        return dec_logits, enc_weights, dec_self_weights, dec_enc_weights
    
# 定义TransformerTrainer类
class TransformerTrainer:
    def __init__(self, model, corpus_loader, dataloader, batch_size=8, learning_rate=0.01, epochs=10, device=None):
        self.model = model
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.corpus_loader = corpus_loader
        self.dataloader = dataloader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")    
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.corpus_loader.word2idx_en['<pad>'])
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
    
    def train(self, checkpoint_path, save_dir):
        start_epoch = 0
        if os.path.isfile(checkpoint_path):
            epoch, _ = self.load_checkpoint(checkpoint_path, self.optimizer, checkpoint_path)
        self.model.to(device)

        # 进行训练
        for epoch in range(start_epoch, self.epochs):
            epoch_loss = 0.0
            with tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f'Epoch {epoch+1}/100', unit='batch') as progress_bar:
                for i, (enc_inputs, dec_inputs, target_batch) in progress_bar:
                    enc_inputs, dec_inputs, target_batch = enc_inputs.to(device), dec_inputs.to(device), target_batch.to(device)
                    self.optimizer.zero_grad()
                    outputs, _, _, _ = self.model(enc_inputs, dec_inputs)
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_batch.view(-1))
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    progress_bar.set_postfix(loss=f'{loss.item():.6f}')
            average_epoch_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch: {epoch + 1:04d} Average Loss: {average_epoch_loss:.6f}")

            if (epoch + 1) % 5 == 0:
                print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
                os.makedirs(save_dir, exist_ok=True)

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss.item()
                }
                formatted_loss = f"{loss.item():.2f}"
                save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}_loss_{formatted_loss}.pth')
                torch.save(checkpoint, save_path)
        
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        print("*******************\n", checkpoint.keys()) 
        epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        return epoch, loss


# 定义GPTDecoderLayer层
class GPTDecoderLayer(nn.Module):
    def __init__(self):
        super(GPTDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()  # 多头自注意力层
        self.feed_forward = PosByPosFeedForwardNet()  # 位置前馈神经网络层
        self.norm1 = nn.LayerNorm(d_embedding)  # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_embedding)  # 第二个层归一化

    def forward(self, dec_inputs, attn_mask=None):
        # 使用多头自注意力处理输入
        attn_output, _ = self.self_attn(dec_inputs, dec_inputs, dec_inputs, attn_mask)
        # 将注意力输出与输入相加并进行第一个层归一化
        norm1_outputs = self.norm1(dec_inputs + attn_output)
        # 将归一化后的输出输入到位置前馈神经网络
        ff_outputs = self.feed_forward(norm1_outputs)
        # 将前馈神经网络输出与第一次归一化后的输出相加并进行第二个层归一化
        dec_outputs = self.norm2(norm1_outputs + ff_outputs)
        return dec_outputs
 
 
#  定义GPTDecoder解码器类
class GPTDecoder(nn.Module):
    def __init__(self, corpus):
        super(GPTDecoder, self).__init__()
        self.src_emb = nn.Embedding(corpus.vocab_size, d_embedding)  # 词嵌入层（参数为词典维度）
        self.pos_emb = nn.Embedding(corpus.seq_len, d_embedding)  # 位置编码层（参数为序列长度）        
        self.layers = nn.ModuleList([GPTDecoderLayer() for _ in range(n_decoder_layers)]) # 初始化N个解码器层

    def forward(self, dec_inputs):        
        positions = torch.arange(len(dec_inputs), device=dec_inputs.device).unsqueeze(-1) # 位置信息        
        inputs_embedding = self.src_emb(dec_inputs) + self.pos_emb(positions) # 词嵌入与位置编码相加        
        attn_mask = subsequent_atten_mask(inputs_embedding).to(dec_inputs.device) # 生成自注意力掩码
        dec_outputs =  inputs_embedding # 初始化解码器输入，这是第一层解码器层的输入      
        for layer in self.layers:
            # 每个解码器层接收前一层的输出作为输入，并生成新的输出
            # 对于第一层解码器层，其输入是dec_outputs，即词嵌入和位置编码的和
            # 对于后续的解码器层，其输入是前一层解码器层的输出            
            dec_outputs = layer(dec_outputs, attn_mask) # 将输入数据传递给解码器层
        return dec_outputs # 返回最后一个解码器层的输出，作为整个解码器的输出

 
#  定义GPT类
class GPT(nn.Module):
    def __init__(self, corpus):
        super(GPT, self).__init__()
        self.corpus = corpus
        self.decoder = GPTDecoder(corpus) # 解码器，用于学习文本生成能力
        self.projection = nn.Linear(d_embedding, corpus.vocab_size)  # 全连接层，输出预测结果
    
    # 将输入数据传递给解码器，经过全连接层后再输出预测值
    def forward(self, dec_inputs):        
        dec_outputs = self.decoder(dec_inputs) # 将输入数据传递给解码器
        logits = self.projection(dec_outputs) # 传递给全连接层以生成预测
        return logits #返回预测结果
    
    def decode(self, input_str, strategy='greedy', **kwargs):
        if strategy == 'greedy': # 贪心解码函数
            return greedy_search(self, input_str, **kwargs)
        elif strategy == 'beam_search': # 集束解码函数
            return beam_search(self, input_str, **kwargs)
        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}")
        

   
#  定义GPTTrainer类
class GPTTrainer:
    def __init__(self, model, corpus, batch_size=24, learning_rate=0.01, epochs=10, device=None):
        self.model = model
        self.corpus = corpus
        self.vocab_size = corpus.vocab_size
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")        
        self.criterion = nn.CrossEntropyLoss(ignore_index=corpus.vocab["<pad>"])
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
    
    def train(self, save_dir):
        self.model.to(self.device)
        # 创建一个整体的 tqdm 进度条，表示 epoch 的进度
        with tqdm(total=self.epochs, desc='Training', unit='epoch') as progress_bar:
            for epoch in range(self.epochs):
                self.optimizer.zero_grad()
                dec_inputs, target_batch = self.corpus.make_batch(self.batch_size)
                dec_inputs, target_batch = dec_inputs.to(self.device), target_batch.to(self.device)
                outputs = self.model(dec_inputs)
                loss = self.criterion(outputs.view(-1, self.corpus.vocab_size), target_batch.view(-1))

                loss.backward()
                self.optimizer.step()

                if (epoch + 1) % 500 == 0:
                    print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
                    os.makedirs(save_dir, exist_ok=True)

                    formatted_loss = f"{loss.item():.2f}"
                    save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}_loss_{formatted_loss}.pth')
                    torch.save(self.model.state_dict(), save_path)

                # 更新进度条
                progress_bar.set_postfix(loss=f'{loss.item():.6f}')
                progress_bar.update(1)