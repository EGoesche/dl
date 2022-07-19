import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        self.min_valid_loss = float('inf') # our addition

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation.
        # -update weights
        # -return the loss

        # Implementations
        # zero the parameter gradients
        self._optim.zero_grad()

        # forward + backward + optimize
        output = self._model(x)
        y = y.float()
        loss = self._crit(output, y)
        loss.backward()
        self._optim.step()
        return loss

        
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions

        self._model.eval()
        output = self._model(x)
        y = y.float()
        loss = self._crit(output, y)
        return loss
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it

        self._model.train(True)
        running_loss = 0.
        last_loss = 0.
        # If you want, intra-epoch reporting, alter the for loop with enumerate and print it every x input
        for pair in self._train_dl:  # data-label pair
            if self._cuda:  # or -> if t.cuda.is_available()
                pair = pair.to('cuda')
            # Every data instance is an input + label pair
            input, label = pair

            # Compute the loss and its gradients
            loss = self.train_step(input, label)

            # Gather data and report
            running_loss += loss.item()

        self._model.train(False) # This may not be needed. But in general, if training is over, this is done.
        train_loss = running_loss / len(self._train_dl)  #Average loss for one epoch
        return train_loss
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #
        self._model.eval()
        running_loss = 0.
        with t.no_grad():
            for pair in self._val_test_dl:  # data-label pair
                if self._cuda:  # or -> if t.cuda.is_available()
                    pair = pair.to('cuda')
                input, label = pair
                loss = self.val_test_step(input, label)
                running_loss += loss.item()
            valid_loss = running_loss / len(self._val_test_dl)
            print(f'Validation Loss: {valid_loss}')
            if valid_loss < self.min_valid_loss:
                print(f'Validation Loss Decreased({self.min_valid_loss:.6f}--->{valid_loss:.6f})')
                self.min_valid_loss = valid_loss

            return valid_loss
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        valid_losses = []
        epoch_counter = 0

        # Increases by 1 if there's no progress after an epoch and a subsequent validation loss.
        # Resets to 0 when progress happens
        patience_counter = 0

        while True:
      
            # stop by epoch number
            # train for an epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            train_losses.append(self.train_epoch())
            valid_losses.append(self.val_test())

            # Save the model. You may want to do that every X epochs rather than after every epoch
            self.save_checkpoint(epoch_counter)

            # Update patience counter according to the latest validation loss
            # If last valid_loss is not the min, it implies loss has not decreased
            if valid_losses[-1] > self.min_valid_loss:
                patience_counter += 1
            else:
                patience_counter = 0

            # Check if our patience is over or epochs are done
            if self._early_stopping_patience == patience_counter or epoch_counter == epochs:
                break
            epoch_counter += 1
        print("Last epoch tried: " + str(epoch_counter))
        return train_losses, valid_losses
                    
        
        
        
