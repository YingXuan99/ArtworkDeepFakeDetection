import tensorflow as tf

class UnfreezeCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, base_model, unfreeze_epoch, unfreeze_block, new_lr=1e-5):
        super().__init__()
        self.model = model
        self.base_model = base_model
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_block = unfreeze_block
        self.new_lr = new_lr
        self.unfrozen = False
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch and not self.unfrozen:
            print(f"\nEpoch {epoch}: Unfreezing blocks 5, 6, 7 and setting LR to {self.new_lr}")
            
            # Unfreeze specified blocks
            for layer in self.base_model.layers:
                for block_name in self.unfreeze_block:
                    if block_name in layer.name:
                        layer.trainable = True
                        break
            
            K = tf.keras.backend
            K.set_value(self.model.optimizer.learning_rate, self.new_lr)
            
            self.unfrozen = True
            print(f"Model unfrozen and learning rate updated to {self.new_lr}")