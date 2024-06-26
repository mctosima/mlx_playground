# Run Image Classification on Apple Silicon (Mac)

## Requirements and Installation
1. Apple Silicon Devices
2. I used Python 3.10
3. Install the requirements using
    ```
    pip install -r requirements.txt
    ```
4. Activate the environment and Run `train.py`

## Dataset
The dataset is available at [Kaggle - Recyclable and household dataset](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification). However, I have already create a script to download the dataset inside the code.

## Models
I'm using the resnet model for MLX. Obtained from [this link](https://github.com/Aavache/mlx-resnet)

## Run Log
```bash
Unique labels: [0 1 2 3 4]
Num classes: 5
== Epoch: 0 ==
Epoch 1/10: 100%|███████████████████████████████████████████████████████████████████████| 31/31 [00:44<00:00,  1.44s/batch]
Loss: 1.8483996391296387 | Time taken: 0:00:44.810968
Validation Accuracy: 0.0 | Time taken: 0:00:00.637666
== Epoch: 1 ==
Epoch 2/10: 100%|███████████████████████████████████████████████████████████████████████| 31/31 [00:46<00:00,  1.49s/batch]
Loss: 1.6912568807601929 | Time taken: 0:00:46.330493
Validation Accuracy: 0.0 | Time taken: 0:00:00.499317
== Epoch: 2 ==
Epoch 3/10: 100%|███████████████████████████████████████████████████████████████████████| 31/31 [00:45<00:00,  1.47s/batch]
Loss: 1.6309419870376587 | Time taken: 0:00:45.703442
Validation Accuracy: 0.0 | Time taken: 0:00:00.506367
== Epoch: 3 ==
Epoch 4/10: 100%|███████████████████████████████████████████████████████████████████████| 31/31 [00:46<00:00,  1.50s/batch]
Loss: 1.5019124746322632 | Time taken: 0:00:46.492178
Validation Accuracy: 0.03125 | Time taken: 0:00:00.497205
== Epoch: 4 ==
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████████| 31/31 [00:44<00:00,  1.44s/batch]
Loss: 1.4747289419174194 | Time taken: 0:00:44.692103
Validation Accuracy: 0.046875 | Time taken: 0:00:00.489129
== Epoch: 5 ==
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████████| 31/31 [00:45<00:00,  1.46s/batch]
Loss: 1.4578289985656738 | Time taken: 0:00:45.401874
Validation Accuracy: 0.0625 | Time taken: 0:00:01.913301
== Epoch: 6 ==
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████████| 31/31 [00:51<00:00,  1.66s/batch]
Loss: 1.4244872331619263 | Time taken: 0:00:51.386369
Validation Accuracy: 0.890625 | Time taken: 0:00:00.495851
== Epoch: 7 ==
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████████| 31/31 [00:45<00:00,  1.48s/batch]
Loss: 1.4477771520614624 | Time taken: 0:00:45.932113
Validation Accuracy: 0.296875 | Time taken: 0:00:00.894764
== Epoch: 8 ==
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████████| 31/31 [00:44<00:00,  1.45s/batch]
Loss: 1.332722783088684 | Time taken: 0:00:44.875769
Validation Accuracy: 0.109375 | Time taken: 0:00:00.473993
== Epoch: 9 ==
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████████| 31/31 [00:45<00:00,  1.47s/batch]
Loss: 1.522979497909546 | Time taken: 0:00:45.430870
Validation Accuracy: 0.96875 | Time taken: 0:00:00.503598
```