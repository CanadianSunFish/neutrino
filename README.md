
# Current Progress

### Tuning Hyperparameters
The current issue seems to be computational time, as discovering the best batch size with training data of size ~250,000 is very slow. Hopefully the batch size determined to be optimal will be something larger than 256, since everything below this increases the computational time up to unreasonably long. (i.e 5-6 minutes of training with batch size 64) 

Once this is resolved we will begin working on searching for optimal learning rate. An array of likely candidates will be created, then we will iterate over the array training the model anew each time, then determine which is optimal.

The issue after this might be model architecture. The current model is 9 layers with relu activation going from 512 nodes and halving down to 2. Then the output node has linear activation. Our hyperparameters are being tuned based on this architecture, however, we will likely do some random search to find more optimal model architecture. This may result in us discovering that other tuned hyperparameters are not compatible with this new architecture. Then we will start fresh to find a new optimizer, batch size, and learning rate. Hopefully after this the model will be sufficiently accurate.  
