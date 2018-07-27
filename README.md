# chexnet implementation
This project an implementation of the [ChexNet paper](https://arxiv.org/abs/1711.05225) that predicts the 14 common pathologies associated with chest x-rays. Data used to train the model consists of 100,000 chest x-ray images (see the full dataset [here](https://nihcc.app.box.com/v/ChestXray-NIHCC)). 

This implementation achieved average AUC of 0.77 across 14 classes compared to AUC of 0.84 reported in the original paper. The reason for the lower performance may be due to the differences in which the learning rate is adjusted. In the original paper, learning rate is decayed by a factor of 10 whenever the validation loss plateaus after an epoch. In this implementation, a step decay method is used. However, I do not want to re-train this beast because I have a puny laptop...

# resources
For other great implementations of ChexNet, please check out:\
https://github.com/jrzech/reproduce-chexnet (PyTorch)\
https://github.com/brucechou1983/CheXNet-Keras (Keras)\

As well as an implementation of DenseNet using Keras:\
https://github.com/flyyufelix/DenseNet-Keras\

For a critical discussion of ChexNet, please read this excellent review by Luke Oakden-Rayner:
https://lukeoakdenrayner.wordpress.com/2018/01/24/chexnet-an-in-depth-review/
