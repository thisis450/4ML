#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import os
# 定义氨基酸字母表和映射
AMINO_ACID_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY-'  # 20个标准氨基酸
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACID_ALPHABET)}  # 字符到索引的映射
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACID_ALPHABET)}  # 索引到字符的映射
END_IDX = len(AMINO_ACID_ALPHABET)  # 结束符的特殊索引
ALL_IDX = list(AA_TO_IDX.values()) + [END_IDX]  # 包含结束符的索引
# 数据集文件路径
data_file_path = './MSA_nat_with_annotation.faa'

# 读取并处理蛋白质序列
def load_protein_data(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):  # 每两个行一组
            sequence = lines[i + 1].strip()  # 获取蛋白质序列
            # 合并多行蛋白质序列（处理换行符）
            full_sequence = ''.join(sequence.split())  # 移除所有换行符
            sequences.append(full_sequence)
    return sequences

# 将氨基酸序列转换为模型输入的张量
def amino_acid_to_tensor(sequence):
    """将氨基酸序列转换为模型输入的张量"""
    tensor = torch.tensor([AA_TO_IDX.get(aa, END_IDX) for aa in sequence])  # 如果是 gap 使用 END_IDX
    return tensor.unsqueeze(0)  # 返回一个 batch 的维度

# 加载并处理数据
protein_sequences = load_protein_data(data_file_path)

# 将所有蛋白质序列转换为张量
sequence_tensors = [amino_acid_to_tensor(seq) for seq in protein_sequences]

# 打印前几个样本的长度和内容
for i in range(3):
    print(f"Sample {i + 1}:")
    print(f"Sequence Length: {sequence_tensors[i].size(1)}")
    print(f"Sequence (first 30 chars): {protein_sequences[i][:30]}")
    print("="*50)


# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义生成器（Generator）模型
class ProteinSequenceGenerator(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, max_seq_len):
        super(ProteinSequenceGenerator, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)  # 输入维度和嵌入维度
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # 输出为氨基酸的数量（包括结束符）

        self.max_seq_len = max_seq_len  # 生成序列的最大长度

    def forward(self, x):
        x = self.embedding(x)  # 将输入序列映射到低维空间
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch_size, seq_len, hidden_dim]
        out = self.fc(lstm_out)  # out shape: [batch_size, seq_len, output_dim]
        return out

    def generate_sequence(self, seed, max_seq_len):
        """
        生成蛋白质序列
        seed: 初始输入（通常是一个随机的氨基酸索引）
        max_seq_len: 生成序列的最大长度
        """
        with torch.no_grad():
            self.eval()
            generated_sequence = seed  # 初始种子
            for _ in range(max_seq_len - seed.size(1)):  # 生成最大序列长度
                output = self.forward(generated_sequence)
                next_idx = torch.argmax(output[:, -1, :], dim=-1)  # 选择概率最大的氨基酸
                next_one_hot = torch.zeros(generated_sequence.size(0), 1, len(ALL_IDX)).scatter_(2, next_idx.unsqueeze(-1).unsqueeze(1), 1)
                
                # 调整维度使得可以拼接
                generated_sequence = generated_sequence.unsqueeze(2)  # 将 generated_sequence 扩展为三维张量 [batch_size, seq_len, 1]
                generated_sequence = torch.cat((generated_sequence, next_one_hot), dim=1)  # 在序列的末尾拼接 next_one_hot

                # 如果生成的序列包含了 END_IDX，则停止生成
                if next_idx.item() == END_IDX:
                    break

            return generated_sequence

# 定义判别器（Discriminator）模型
class ProteinSequenceDiscriminator(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(ProteinSequenceDiscriminator, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)  # 输入维度和嵌入维度
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # 输出为真假判断

    def forward(self, x):
        x = self.embedding(x)  # 将输入序列映射到低维空间
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch_size, seq_len, hidden_dim]
        out = self.fc(lstm_out[:, -1, :])  # 只取最后一层输出进行分类
        return torch.sigmoid(out)

# 定义生成器和判别器的优化器和损失函数
generator = ProteinSequenceGenerator(len(ALL_IDX), embedding_dim=64, hidden_dim=128, output_dim=len(ALL_IDX), max_seq_len=100)
discriminator = ProteinSequenceDiscriminator(len(ALL_IDX), embedding_dim=64, hidden_dim=128)

criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 生成一个种子序列并将其转换为张量
def amino_acid_to_tensor(sequence):
    """将氨基酸序列转换为模型输入的张量"""
    tensor = torch.tensor([AA_TO_IDX.get(aa, END_IDX) for aa in sequence])  # 如果是 gap 使用 END_IDX
    return tensor.unsqueeze(0)  # 返回一个 batch 的维度

# 示例：训练生成器和判别器
for epoch in range(10000):
    # 生成真实数据（用一些实际的蛋白质序列）
    real_sequence = amino_acid_to_tensor("M" * 100)  # 示例的真实序列，假设有 100 个氨基酸
    batch_size = real_sequence.size(0)
    
    # 生成假的数据（使用生成器）
    noise = torch.randint(0, len(ALL_IDX), (batch_size, 100)).long()  # 随机噪声，种子为随机索引
    fake_sequence = generator.generate_sequence(noise, max_seq_len=100)

    # 判别器训练
    optimizer_D.zero_grad()
    
    real_output = discriminator(real_sequence)
    fake_output = discriminator(fake_sequence.detach())  # 不计算梯度
    
    real_label = torch.ones(batch_size, 1)
    fake_label = torch.zeros(batch_size, 1)
    
    real_loss = criterion(real_output, real_label)
    fake_loss = criterion(fake_output, fake_label)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    # 生成器训练
    optimizer_G.zero_grad()
    
    output = discriminator(fake_sequence)
    g_loss = criterion(output, real_label)  # 目标是让生成的序列被判别器判为真实
    g_loss.backward()
    optimizer_G.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/10000], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# 示例：生成一个蛋白质序列
noise = torch.randint(0, len(ALL_IDX), (1, 100)).long()  # 随机生成一个种子
generated_sequence = generator.generate_sequence(noise, max_seq_len=100)

# 将生成的张量转换回氨基酸序列
def tensor_to_amino_acid(tensor):
    return ''.join([IDX_TO_AA[idx.item()] if idx.item() < len(AMINO_ACID_ALPHABET) else '-' for idx in tensor.squeeze()])

generated_protein_sequence = tensor_to_amino_acid(generated_sequence)
print(f"Generated Protein Sequence: {generated_protein_sequence}")



# In[ ]:




