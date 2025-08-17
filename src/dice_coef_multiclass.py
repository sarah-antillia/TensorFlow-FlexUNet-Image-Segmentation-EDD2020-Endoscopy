#
# dice_coef for multi-classe 
#
import tensorflow.keras.backend as K

# Please see:
# Multiclass segmentation for different loss functions(Dice loss, Focal loss, Total loss = (Summation of Dice and focal loss)) in Tensorflow
# https://medium.com/@mb16biswas/multiclass-segmentation-for-different-loss-functions-dice-loss-focal-loss-total-loss-summation-455178517cea

# https://gist.github.com/sohiniroych/68ce46adfae0400acc5fe833d96f6464#file-loss_functions-py

# https://stackoverflow.com/questions/61488732/how-calculate-the-dice-coefficient-for-multi-class-segmentation-task-using-pytho

# https://www.kaggle.com/code/mb16biswas/multiclass-segmentation-for-diff-loss-functions


# Dice_coef for multi-class segmentation
def dice_coef_multiclass(y_true, y_pred, smooth=1):
    """
    Dice coefficient for multi-class segmentation.
    Args:
        y_true: Ground truth tensor (one-hot encoded). Shape: (batch, height, width, num_classes)
        y_pred: Prediction tensor (probabilities). Shape: (batch, height, width, num_classes)
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Dice coefficient.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def dice_loss_multiclass(y_true, y_pred, smooth=1):
    """
    Dice loss, which can be minimized during training.
    """
    return 1 - dice_coef_multiclass(y_true, y_pred, smooth)