
# Current Progress

### Tuning Hyperparameters
The current issue seems to be computational time, as discovering the best batch size with training data of size ~1.5 million is very slow. Hopefully the batch size determined to be optimal will be something larger than 1000, since everything below this increases the computational time up to unreasonably long. (i.e 5-6 minutes per epoch with batch size 64) 

Once this is resolved we will begin working on searching for optimal learning rate. An array of likely candidates will be created, then we will iterate over the array training the model anew each time, then determine which is optimal.

The issue after this might be model architecture. The current model is 9 layers with relu activation going from 512 nodes and halving down to 2. Then the output node has linear activation. Our hyperparameters are being tuned based on this architecture, however, we will likely do some random search to find more optimal model architecture. This may result in us discovering that other tuned hyperparameters are not compatible with this new architecture. Then we will start fresh to find a new optimizer, batch size, and learning rate. Hopefully after this the model will be sufficiently accurate.  

### Collecting Event Data
Another issue may be event data. Training on under 100,000 events would yield mean absolute error of around 0.17+, while pushing this up to 1,000,000 without working on hyperparameter tuning dropped this error to 0.162+.

Again computational time will be a large issue.

### Computational Time
Seems valuable to discuss as it has been mentioned many times already. Finding ways to reduce the training time while trying to optimize the model will be necessary. This can by done through a few techniques. Note: These are only the first that come to mind.
- Saving model architectures in a state of half-accuracy, then loading these models later when using the same architecture but different hyperparameters
  - Larger models can be run for 1 epoch with new hyperparameters and evaluated immediatly
- Utilizing well know tuning packages through keras, scikit-learn, pytorch, or others
  - Larger teams who have stronger background in computer science will notable have better code
