import sys
import glob
import cnn_model as cnn
from cilutil import resizing

# Train CNN
cnn.main()

# Upsample predictions for both training and test set
UPSAMPLE = True
if UPSAMPLE:
    training_filenames = glob.glob("../results/CNN_Output/training/*/*.png")
    test_filenames = glob.glob("../results/CNN_Output/test/*/*.png")
    resizing.upsample_training(training_filenames)
    resizing.upsample_test(test_filenames)
