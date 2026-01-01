import tensorflow as tf

def check_gpu():
    """
    Checks and prints available GPUs.
    """
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    return len(gpus) > 0
