# 独立模型训练脚本（高准确率版）
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# 固定随机种子
np.random.seed(42)

# 1. 读取独立数据集
df = pd.read_csv("metro_nlp_dataset.csv", encoding="utf-8-sig")
texts = df["query_text"].values
start_labels = df["start_station"].values

# 站点列表
all_stations = df["start_station"].unique().tolist()
station2idx = {s: i for i, s in enumerate(all_stations)}

# 2. 8:2 训练测试分割
X_train, X_test, y_train, y_test = train_test_split(
    texts, start_labels, test_size=0.2, random_state=42
)

# 3. 文本预处理
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
tokenizer.fit_on_texts(X_train)
max_len = 20

# 序列编码
train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

# 标签编码
y_train_idx = np.array([station2idx[i] for i in y_train])
y_test_idx = np.array([station2idx[i] for i in y_test])

# 4. 优化版CNN模型（防过拟合+提升精度）
model = Sequential([
    Embedding(len(tokenizer.word_index)+1, 128, input_length=max_len),
    Conv1D(64, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.3),  # 防过拟合!!!
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(all_stations), activation='softmax')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 早停法（防止过拟合!!!）
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 5. 训练模型
print("\n🚀 开始训练CNN模型...")
model.fit(
    train_seq, y_train_idx,
    validation_data=(test_seq, y_test_idx),
    epochs=15,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# 6. 评估模型
test_loss, test_acc = model.evaluate(test_seq, y_test_idx, verbose=0)
print(f"\n🎯 最终测试集准确率：{test_acc:.2%}")

# 7. 保存模型（新版格式）
model.save("metro_cnn_model.keras")
with open("metro_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("metro_stations.pkl", "wb") as f:
    pickle.dump(all_stations, f)

print("\n" + "="*60)
print("🏆 模型训练完成！")
print(f"🏅 准确率：{test_acc:.2%}")
print("="*60)