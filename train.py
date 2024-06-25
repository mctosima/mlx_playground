import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from resnet import resnet50
from household_waste import HouseholdWasteDataset
from household_waste import get_transforms
from functools import partial
# from mlx.utils import tree_map


NUM_CLASSES = 30

def data_loader(dataset, batch_size, shuffle=True):
    data_len = len(dataset)
    indices = np.arange(data_len)
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, data_len, batch_size):
        end_idx = min(start_idx + batch_size, data_len)
        batch_indices = indices[start_idx:end_idx]
        batch = [dataset[i] for i in batch_indices]
        
        batch_X = mx.array([item[0] for item in batch])
        batch_y = mx.array([item[1] for item in batch])  # Ensure labels are integers
        
        yield batch_X, batch_y

def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

def train(
    num_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,  
):
    
    # 1. Initiate Model
    model = resnet50(num_classes = NUM_CLASSES)
    mx.eval(model.parameters()) # Initialize model parameters
    
    
    # print(model)
    
    
    # 2. Load Dataset
    train_data = HouseholdWasteDataset(
        root_dir="recyclable-and-household-waste-classification/images/images",
        split="train",
        transform=get_transforms(),
    )
    
    val_data = HouseholdWasteDataset(
        root_dir="recyclable-and-household-waste-classification/images/images",
        split="val",
        transform=get_transforms(),
    )
    
    train_loader = data_loader(train_data, batch_size, shuffle=True)
    val_loader = data_loader(val_data, batch_size, shuffle=False)
    
    # 3. Hyperparameters
    optimizer = optim.Adam(learning_rate = lr)
    
    # 4. Loss function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # 5. Define forward pass step
    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def train_step(X, y):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss
    
    @partial(mx.compile, inputs=model.state)
    def eval_fn(X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)
    
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        
        # Training
        num_batches = len(train_data) // batch_size + int(len(train_data) % batch_size != 0)
        for idx, (batch_X, batch_y) in enumerate (train_loader):
            loss = train_step(batch_X, batch_y)
            mx.eval(model.state)
            
            percent_complete = (idx + 1) / num_batches * 100
            if idx % 100 == 0:
                print(f"Batch: {idx}/{len(train_data)//batch_size}, Loss: {loss}")
            # break # break for training loop
        
        
        for batch_X, batch_y in val_loader:
            accuracy_step = eval_fn(batch_X, batch_y)
            print(f"Validation Accuracy: {accuracy_step}")
            # break # break for validation loop
        
        break # break for epoch loop
    

if __name__ == "__main__":
    train(
        num_epochs=50,
        batch_size=256,
        lr=1e-3,
    )