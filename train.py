import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from resnet import resnet50
from household_waste import HouseholdWasteDataset
from household_waste import get_transforms
from functools import partial
from tqdm import tqdm
from datetime import datetime


def check_num_classes(dataset):
    labels = [item[1] for item in dataset]
    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")
    return len(unique_labels)


def data_loader(dataset, batch_size, shuffle=True, drop_last=False):
    data_len = len(dataset)
    indices = np.arange(data_len)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, data_len, batch_size):
        end_idx = start_idx + batch_size

        if end_idx > data_len:
            if drop_last:
                break
            end_idx = data_len

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

    # 1. Load Dataset
    train_data = HouseholdWasteDataset(
        root_dir="recyclable-and-household-waste-classification/images/images",
        split="train",
        transform=get_transforms(),
        partial=0.2,
    )

    val_data = HouseholdWasteDataset(
        root_dir="recyclable-and-household-waste-classification/images/images",
        split="val",
        transform=get_transforms(),
        partial=0.2,
    )

    num_cls = check_num_classes(train_data)
    print(f"Num classes: {num_cls}")

    # 2. Initiate Model
    model = resnet50(num_classes=num_cls)
    mx.eval(model.parameters())  # Initialize model parameters

    # 3. Hyperparameters
    optimizer = optim.SGD(learning_rate=lr)

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
        train_loader = data_loader(train_data, batch_size, shuffle=True, drop_last=True)
        val_loader = data_loader(val_data, batch_size, shuffle=False, drop_last=False)
        num_batches = len(train_data) // batch_size
        current_time = datetime.now()
        print(f"== Epoch: {epoch} ==")

        # Training
        temp_loss = []
        for idx, (batch_X, batch_y) in enumerate(
            tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                unit="batch",
                total=num_batches,
            )
        ):
            loss = train_step(batch_X, batch_y)
            temp_loss.append(loss)
            mx.eval(model.state)
        final_loss = sum(temp_loss) / len(temp_loss)
        print(
            f"Loss: {final_loss.item()} | Time taken: {datetime.now() - current_time}"
        )

        # Validation
        current_time = datetime.now()
        temp_acc = []
        for idx, (batch_X, batch_y) in enumerate(val_loader):
            accuracy_step = eval_fn(batch_X, batch_y)
            temp_acc.append(accuracy_step)
            break
        final_acc = sum(temp_acc) / len(temp_acc)
        print(
            f"Validation Accuracy: {final_acc.item()} | Time taken: {datetime.now() - current_time}"
        )


if __name__ == "__main__":
    train(
        num_epochs=10,
        batch_size=64,
        lr=1e-3,
    )
