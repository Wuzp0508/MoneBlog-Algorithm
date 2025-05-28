import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, f1_score
import json
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class AttentionLayer(nn.Module):
    """注意力机制层"""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)  # 双向LSTM需要hidden_dim*2

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim*2)
        attention_weights = torch.tanh(self.attention(lstm_output))  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)  # (batch_size, seq_len)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch_size, hidden_dim*2)
        return context_vector, attention_weights


class TextClassifier(nn.Module):
    """带注意力机制的文本分类器"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout=0.3):
        super(TextClassifier, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 双向LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        # 注意力层
        self.attention = AttentionLayer(hidden_dim)
        # 分类层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, return_attention=False):
        embedded = self.embedding(x)  # 嵌入层 (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # LSTM层 (batch_size, seq_len, hidden_dim*2)
        context_vector, attention_weights = self.attention(lstm_out)  # 注意力层
        output = self.fc(context_vector)  # 分类
        if return_attention:
            return output, attention_weights
        return output

# 数据预处理
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len=100):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 将文本转换为索引序列
        text_idx = [self.word_to_idx.get(word, 1) for word in jieba.cut(text)]  # <UNK>的索引设为1
        text_idx = text_idx[:self.max_len]
        text_idx += [0] * (self.max_len - len(text_idx))  # 填充到固定长度
        return {
            'text': torch.tensor(text_idx, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def build_vocab(texts, min_count=2):
    word_count = Counter()
    for text in texts:
        words = jieba.cut(text)
        word_count.update(words)
    vocab = [word for word, count in word_count.items() if count >= min_count]
    vocab = ['<PAD>', '<UNK>'] + vocab
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return word_to_idx


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_layers, embedding_dim, hidden_dim, epochs):
    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            text = batch['text'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 验证
        val_loss, accuracy, f1 = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}\n')
        # 更新学习率
        scheduler.step(val_loss)
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f'Attention_numLayers={num_layers}_embeddingDim={embedding_dim}_hiddenDim={hidden_dim}_best_model.pth')
    print(f'Best Validation F1: {best_f1:.4f}')


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            text = batch['text'].to(device)
            label = batch['label'].to(device)
            output = model(text)
            loss = criterion(output, label)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(classification_report(all_labels, all_preds, target_names=['消极', '中性', '积极']))
    return avg_loss, accuracy, f1


def visualize_attention(text, attention_weights, top_k=5):
    """可视化注意力权重"""
    words = list(jieba.cut(text))
    attention_weights = attention_weights[:len(words)]  # 截断到实际文本长度
    # 获取权重最高的top_k个词
    top_indices = np.argsort(attention_weights)[-top_k:][::-1]
    top_words = [words[i] for i in top_indices]
    top_weights = [attention_weights[i] for i in top_indices]
    # 绘制条形图
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(top_weights)), top_weights)
    plt.xticks(range(len(top_weights)), top_words, rotation=45)
    plt.title(f"注意力权重Top{top_k}")
    plt.xlabel("分词")
    plt.ylabel("注意力权重")
    plt.show()
    # 高亮显示文本
    highlighted_text = []
    for word, weight in zip(words, attention_weights):
        if weight > np.percentile(attention_weights, 75):  # 高亮前25%的词语
            highlighted_text.append(f"\033[1;31m{word}\033[0m")  # 红色高亮
        else:
            highlighted_text.append(word)
    print("高亮关键片段:", "".join(highlighted_text))


def text_classification(text):
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, 'vocab.json'), 'r', encoding="utf-8") as file:
        vocab = json.load(file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dim, hidden_dim, output_dim, num_layers = 256, 256, 3, 1
    model = TextClassifier(len(vocab), embedding_dim, hidden_dim, output_dim, num_layers, dropout=0).to(device)
    model.load_state_dict(torch.load(os.path.join(current_path, f'Attention_numLayers={num_layers}_embeddingDim={embedding_dim}_hiddenDim={hidden_dim}_best_model.pth'), map_location=torch.device('cpu')))
    model.eval()
    idx = [vocab.get(word, 1) for word in jieba.cut(text)]
    tensor = torch.LongTensor(idx).unsqueeze(0).to(device)
    output, attention_weights = model(tensor, return_attention=True)
    pred_label = ['消极', '中性', '积极'][torch.argmax(output)]
    attention_weights = attention_weights.view(-1).detach().cpu().numpy()
    return f"预测结果: {pred_label}"
    visualize_attention(text, attention_weights)


if __name__ == "__main__":
    # 加载数据
    train_df, val_df = pd.read_csv('base_train.csv', encoding='gbk'), pd.read_csv('base_val.csv', encoding='gbk')
    # 提取文本和标签
    train_texts, val_texts = train_df['text'].tolist(), val_df['text'].tolist()
    sentiment_mapping = {'消极': 0, '中性': 1, '积极': 2}
    train_labels, val_labels = train_df['label'].map(sentiment_mapping).tolist(), val_df['label'].map(
        sentiment_mapping).tolist()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 构建词汇表
    vocab = build_vocab(train_texts + val_texts)
    with open('vocab.json', "w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=4)
    with open('vocab.json', "r", encoding="utf-8") as file:
        vocab = json.load(file)
    # 模型参数
    embedding_dim, hidden_dim, output_dim, num_layers = 256, 256, 3, 1
    model = TextClassifier(69656, embedding_dim, hidden_dim, output_dim, num_layers, dropout=0.3).to(device)
    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_texts, train_labels, vocab)
    val_dataset = TextDataset(val_texts, val_labels, vocab)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)
    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    train_flag = False
    if train_flag:
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_layers, embedding_dim, hidden_dim, epochs=20)
    # 加载最佳模型并评估
    model.load_state_dict(torch.load(f'Attention_numLayers={num_layers}_embeddingDim={embedding_dim}_hiddenDim={hidden_dim}_best_model.pth', map_location=device))
    _, test_accuracy, test_f1 = evaluate_model(model, val_loader, criterion, device)
    print(f'Final Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}')
    # 验证集评估+可视化示例
    model.eval()
    sample_texts = ['大病刚好来补补[嘻嘻]//@?痴高文麒:上海同胞好幸福啊[嘻嘻]',
                    '看来悟空也在为创意冥思苦想，各种进入最后总决赛的同学们，加油！努力在这次比赛的平台上展示出自己的创意！[鼓掌][鼓掌]',
                    '嗓子疼死了。要买口罩了[衰]']
    for (i, sample) in enumerate(sample_texts):
        print(f'\nNo.{i + 1} Validation')
        print(f'测试文本： {sample}')
        sample_idx = [vocab.get(word, 1) for word in jieba.cut(sample)]  # 1 是 UNK 的索引
        sample_tensor = torch.LongTensor(sample_idx).unsqueeze(0).to(device)
        # 获取预测和注意力权重
        with torch.no_grad():
            output, attention_weights = model(sample_tensor, return_attention=True)
        pred_label = ['消极', '中性', '积极'][torch.argmax(output)]
        attention_weights = attention_weights.squeeze().cpu().numpy()  # (seq_len, )
        print(f"预测结果: {pred_label}")
        visualize_attention(sample, attention_weights)
    # word = '少奇主动帮助乔珊，因此得知乔珊男友的名字和自己名字的发音完全一样，只是字面不一样。'
    # top_weights = [0.041, 0.041, 0.040, 0.040, 0.035]
    # top_words = ['一样', '完全', '因此', '不', '只']
    # colors = [
    #     (1, 0.6, 0, 1),  # 深黄色
    #     (1, 0.7, 0, 1),  # 稍浅
    #     (1, 0.8, 0, 1),  # 再浅
    #     (1, 0.9, 0, 1),  # 更浅
    #     (1, 1, 0, 1)  # 最浅
    # ]
    # plt.figure(figsize=(10, 3))
    # plt.bar(range(len(top_weights)), top_weights, color=colors)
    # plt.xticks(range(len(top_weights)), top_words)
    # plt.title(f"注意力权重 Top 5 分词")
    # plt.grid(True)
    # plt.xlabel("分词")
    # plt.ylabel("注意力权重", rotation=90)
    # plt.show()
    # # 高亮显示文本
    # f"\033[1;31m{word}\033[0m"
    # highlighted_text = ['少奇主动帮助乔珊，', '\033[1;31m因此\033[0m' ,'得知乔珊男友的名字和自己名字的发音', '\033[1;31m完全一样\033[0m', '，', '\033[1;31m只\033[0m', '是字面', '\033[1;31m不\033[0m', '一样。']
    # print("预测结果： 中性")
    # print("高亮关键片段:", "".join(highlighted_text))