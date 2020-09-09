# Land Use and Land Cover Classification

## Getting started

1.  Clone the repo: `git clone https://github.com/alicsalman1/EuroSAT_Classification.git`

2.  Install dependencies:
    ```
    conda create -n land_classification python=3.7
    conda activate land_classification
    pip install -r requirements.txt
    ```
3. Download the dataset: [EuroSAT](http://madm.dfki.de/files/sentinel/EuroSAT.zip )


## Training
Run the following script:
```
python train.py --data_dir <path_to_dataset_folder> \
    --epochs 75 \
    --lr 0.1 \
    --step_lr 25 \
    --batch_size 128 \
    --weight_decay 5e-4 \
    --log_file training.log
```

## Evaluation
If you only want to evaluate the model, run:
```
python train.py --eval_only \
    --model checkpoints/best_model.pth 
```

This will plot the confusion matrix and print the classification report (precision, recall, f1 score, accuracy).


## Inference

Run the following script:
```
python inference.py images/Forest.jpg --model checkpoints/best_model.pth
```

## Results
The pretrained model can be found [here](checkpoints/best_model.pth).

### Accuracy
|Model|Accuracy|
|:----:|:----:|
ResNet-18|91.95%

### Classification Report

|Class|Precision|Recall|F1-score|
------------ | -------------| -------------| -------------
AnnualCrop      |0.9109  |0.8950  |0.9029
Forest          |0.9729  |0.9875  |0.9801
HerbaVeg        |0.8899  |0.9100  |0.8999
Highway         |0.9005  |0.8375  |0.8679   
Industrial      |0.9251  |0.9575  |0.9410   
Pasture         |0.8839  |0.9325  |0.9075   
PermanentCrop   |0.8433  |0.8475  |0.8454 
Residential     |0.9802  |0.9875  |0.9838  
River           |0.8964  |0.8650  |0.8804  
SeaLake         |0.9924  |0.9750  |0.9836  

### Confusion matrix
![Confusion Matrix](images/confusion_matrix.png)


## Discussions

#### Dataset
1. The dataset was not balanced, so I only used 2000 images from each class (which was the minimum class size between the 10 classes).

2. I used a 80/20 train/test split. Also, I used stratify option while doining the split to make sure the classes in both train and test sets are balanced.

3. It is always good to normalize data for training, so I first computed the mean and standard deviation of each of the RDB channels of the images, which I then used to notrmalize the values to [0, 1].


#### Architecture 
1. I used a ResNet-18 architecture which has been used alot for classification tasks and shown good results. ResNet-50 is also a good choice and may boost the performence, however I was working with a bad gpu (< 2gb) which was not enough to fit and train this model.
2. I added a classification head inorder to output only 10 values (one for each class).
3. The network was pretrained on ImageNet.


#### Training
1. SGD optimizer with Cross Entropy.
2. Initial learning rate of value 0.1, with a scheduler the decays the value by a factor of 10 after each 25 epochs.
3. A weight decay of 5e-4.
4. Batches of size 128.
4. 75 epochs. This was actually too much for fine tunning, so one can train for fewer epochs.