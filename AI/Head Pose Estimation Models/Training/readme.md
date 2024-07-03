# Training pipeline for Headpose model
## Dependencies to install
1. cuda and cudnn installed on system
2. pytorch with gpu support
3. torchvision
4. tqdm
5. tensorboard 
6. numpy
7. pandas
## how to run
first install the package in requirements.txt file preferably in an isolated environment 
```console

pip install -r requirements.txt
```
second get the corresponding variants from

- <a href="https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq">RepVGG models</a> 
- <a href="https://drive.google.com/drive/folders/1FmEwGVxDZ4qC5cEH0PYaY_98LEDz9WhP?usp=drive_link">efficientnet lite models</a> 

### Train RepVGG B1G2
```console

python TrainRepVGGB1g2.py
```
### Train RepVGG variants A0-B1
```console

python TrainRepVGGA0-B1.py
```
### Train Custom RepVGG variants AX-AY-AZ
```console

python TrainRepVGGspecial.py
```
### Train EfficientNet Lite variants 0-3
```console

python TrainRepVGGspecial.py
```

## explanation for the main training python module
This Python script trains a neural network model (SixDRepNet) for head pose estimation using different backbone architectures (e.g., RepVGG) on two datasets: 300W-LP and BIWI. Here's a brief breakdown:

1. Imports necessary libraries and modules including PyTorch, NumPy, and custom-defined modules.
2. Defines a custom loss function called GeodesicLoss, which calculates the geodesic distance between two rotation matrices.
3. Defines the main function, which orchestrates the training process.
4. Loads pretrained backbone models for different architectures.
5. Prepares datasets for training and testing, including data augmentation and normalization.
6. Sets up training parameters such as batch size, learning rate, and number of epochs.
7. Trains the model for the specified number of epochs, evaluating performance on the validation dataset after each epoch.
8. Computes mean absolute errors (MAE) for yaw, pitch, and roll angles between predicted and ground truth poses.
9. Selects the best model based on the lowest MAE.
10. Saves the best model and its performance metrics.
11. Closes the TensorBoard writer and prints the results.

## Results
### RepVGG A0-B1 Models
![image]("./repvgggraph.png")
### Custom RepVGG AX-AY-AZ Models
![image]("./axazgraph.png")
### EfficientNet Lite 0-3 Models
![image]("./effgraph.png")

## PreTrained Models
- <a href="https://drive.google.com/drive/folders/16ynMygSYC5ysvucNvzVCgJ81QCjO3wPC?usp=sharing">RepVGG A0-B1 Models</a>
- <a href="https://drive.google.com/drive/folders/1yFKmrGoyEFrR_T9ZfZJNLPWP0XXHnSCM?usp=drive_link">Custom RepVGG AX-AY-AZ Models</a>
- <a href="https://drive.google.com/drive/folders/1yFKmrGoyEFrR_T9ZfZJNLPWP0XXHnSCM?usp=drive_link">EfficientNet Lite Models</a>
