import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DecisionTree.DecisionTreeClassifierGPU import DecisionTreeClassifierGPU
from tensorflow.keras.datasets import mnist
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(x_train[:10000], y_train[:10000], test_size=0.2, random_state=41)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, X_test = torch.tensor(X_train).to(device), torch.tensor(X_test).to(device)
Y_train, Y_test = torch.tensor(Y_train).to(device), torch.tensor(Y_test).to(device)

classifier_gpu = DecisionTreeClassifierGPU(min_samples_split=3, max_depth=10)

start_time = time.time()
classifier_gpu.fit(X_train.cpu().numpy(), Y_train.cpu().numpy())
end_time = time.time()
print(f"GPU Model Training Time: {end_time - start_time:.2f} seconds")

Y_pred_gpu = classifier_gpu.predict(X_test.cpu().numpy())

accuracy = accuracy_score(Y_test.cpu().numpy(), Y_pred_gpu)
print(f"GPU Model Accuracy: {accuracy:.4f}")

