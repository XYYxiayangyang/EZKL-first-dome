# EZKL DEMO README

This demo demonstrates how to train a samll model using PyTorch, convert it into a zero-knowledge (ZK) circuit using EZKL, and deploy the ZK verifier on Ethereum. You will learn the process of training a neural network, converting it to a ZK model, generating proofs, and verifying them on the Ethereum blockchain.

## Learning Objectives

- Learn how to train a neural network model using PyTorch to perform classification.
- Convert the trained model into a zero-knowledge (ZK) circuit using EZKL.
- Deploy the ZK verifier on the Ethereum blockchain using Solidity and Remix.

## Prerequisites

- Python 3.x
- PyTorch
- EZKL
- Solidity (version 0.8.20)

## Step 1: Training the Model

We will use the **Iris dataset**, a classic dataset for classification tasks. The dataset contains measurements of iris flowers from three species.

### 1.1 Import Dependencies

Start by importing the necessary dependencies:

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm
```

### 1.2 Inspect the Iris Dataset

The dataset consists of 4 features (sepal length, sepal width, petal length, and petal width) and a target variable representing the iris species.

```python
iris = load_iris()
dataset = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target']
)
dataset
```

### 1.3 Define the Neural Network Model

Here is a simple fully connected neural network with 3 layers and ReLU activation functions.

```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

model = Model()
```

### 1.4 Train the Model

We will use **Cross-Entropy Loss** and **Stochastic Gradient Descent (SGD)** for optimization. The model is trained for 800 epochs.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
EPOCHS = 800

train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y.values).long())
test_y = Variable(torch.Tensor(test_y.values).long())

for epoch in tqdm.trange(EPOCHS):
    predicted_y = model(train_X)
    loss = loss_fn(predicted_y, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        y_pred = model(test_X)
        correct = (torch.argmax(y_pred, dim=1) == test_y).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()
```

### 1.5 Evaluate the Model

Plot the accuracy and loss over training epochs.

```python
import matplotlib.pyplot as plt

plt.style.use('ggplot')

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)
ax1.plot(accuracy_list)
ax1.set_ylabel("Accuracy")
ax2.plot(loss_list)
ax2.set_ylabel("Loss")
ax2.set_xlabel("epochs")
```

### 1.6 Exercise: Improve the Model

- Try different architectures (e.g., more layers, different activation functions).
- Experiment with optimizers like Adam or RMSprop.
- Adjust hyperparameters like learning rate and batch size.

## Step 2: ZK the Neural Network

Now that we have trained the neural network, we will convert it into a **ZK circuit** using EZKL.

### 2.1 Install EZKL

If you're using Colab, install EZKL with the following:

```bash
!pip install ezkl
!pip install onnx
```

### 2.2 Export the Model to ONNX

First, export the trained model to an **ONNX** file format.

```python
x = test_X[0].reshape(1, 4)
model.eval()

torch.onnx.export(model, x, model_path, export_params=True, opset_version=10, 
                  input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
```

### 2.3 Calibrate and Optimize the Model

Generate the settings file and calibrate the ZK model for efficient proof generation.

```python
res = ezkl.gen_settings()
cal_data = dict(input_data=test_X.flatten().tolist())
json.dump(data, open(cal_data_path, 'w'))

res = await ezkl.calibrate_settings(target="resources", max_logrows=12, scales=[2])
```

### 2.4 Compile the Circuit

Now, compile the model into a ZK circuit.

```python
res = ezkl.compile_circuit()
assert res == True
```

### 2.5 Generate Proof and Verify

Generate a proof for the model and verify it using EZKL.

```python
witness_path = os.path.join('witness.json')
res = await ezkl.gen_witness()
proof_path = os.path.join('proof.json')
proof = ezkl.prove(proof_type="single", proof_path=proof_path)
assert os.path.isfile(proof_path)

res = ezkl.verify()
assert res == True
print("verified")
```

## Step 3: Deploying the Verifier

Next, we'll deploy the ZK verifier to the Ethereum blockchain.

### 3.1 Install Solidity Compiler

Make sure you have the correct version of **Solidity** installed.

```bash
!solc-select install 0.8.20
!solc-select use 0.8.20
!solc --version
```

### 3.2 Create the Verifier Contract

Using EZKL, generate the Solidity code for the ZK verifier.

```python
res = await ezkl.create_evm_verifier(sol_code_path=sol_code_path, abi_path=abi_path)
assert res == True
assert os.path.isfile(sol_code_path)
```

### 3.3 Deploy the Verifier

Now, deploy the contract to **Remix**:

1. Go to [Remix](https://remix.ethereum.org).
2. Create a new file and paste the contents of the `Verifier.sol`.
3. Compile and deploy the contract to a test network.
4. Use the `formatted_output` and `proof["proof"]` to test the verifier.

```python
formatted_output = "[ ... ]"  # Copy the formatted output from the earlier code
```

### 3.4 Verify the Proof Onchain

Verify if the proof is valid by using the values from the `formatted_output` and the proof generated earlier.

```python
# Copy values into Remix and see if they verify
print("pubInputs: ", formatted_output)
print("proof: ", proof["proof"])
```

## Conclusion

You have successfully:

1. Trained a toy neural network.
2. Converted the neural network into a ZK circuit using EZKL.
3. Deployed a ZK verifier on the Ethereum blockchain using Solidity.

## References

- [EZKL Documentation](https://github.com/aztecprotocol/ezkl)
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
