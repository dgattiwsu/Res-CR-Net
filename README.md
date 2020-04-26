# Res-CR-Net

Res-CR-Net, a residual network with a novel architecture optimized for the semantic segmentation of microscopy images.

Res-CR-Net is a neural network featuring a novel FCN architecture, with very good performance in multiclass segmentation tasks of both electron (gray scale, 1 channel) and light microscopy (rgb color, 3 channels) images of relevance in the analysis of pathology specimens. Res-CR-Net offers some advantages with respect to other networks inspired to an encoder-decoder architecture, as it is completely modular, with residual blocks that can be proliferated as needed, and it can process images of any size and shape without changing layers size and operations. Res-CR-Net can be particularly effective in segmentation tasks of biological images, where the labeling of ground truth classes is laborious, and thus the number of annotated/labeled images in the training set is small.

Res-CR-Net combines two types of residual blocks:

CONV RES BLOCK. The traditional U-Net backbone architecture, with its the encoder-decoder paradigm, is replaced by a series of modified residual blocks, each consisting of three parallel branches of separable + atrous convolutions with different dilation rates, that produce feature maps with the same spatial dimensions as the original image. The rationale for using multiple-scale layers is to extract object features at various receptive field scales. Res-CR-Net offers the option of concatenating or adding the parallel branches inside the residual block before adding them to the shortcut connection. In our test, concatenation produced the best result. A Spatial Dropout layer follows each residual block. A slightly modified STEM block processes the initial input to the network. n CONV RES BLOCKS can be concatenated.

LSTM RES BLOCK. A new type of residual block features a residual path with two orthogonal bidirectional 2D convolutional Long Short Term Memory (LSTM) layers. For this purpose, the feature map 4D tensor emerging from the previous layer first undergoes a virtual dimension expansion to 5D tensor (i.e. from [4,260,400,3] [batch size, rows, columns, number of classes] to [4,260,400,3,1]). In this case the 2D LSTM layer treats 260 consecutive tensor slices of dimensions [400,3,1] as the input data at each iteration. Each slice is convolved with a single filter of kernel size [3,3] with ‘same’ padding, and returns a slice of the exact same dimension. In one-direction mode the LSTM layer returns a tensor of dimensions [4,260,400,3,1]. In bidirectional mode it returns a tensor of dimensions [4,260,400,3,2]. The intuition behind using a convolutional LSTM layer for this operation lies in the fact that adjacent image rows share most features, and image objects often contain some level of symmetry that can be properly memorized in the LSTM unit. Since the same intuition applies also to image columns, the expanded feature map of dimensions [4,260,400,3,1] from the earlier part of the network is transposed in the 2nd and 3rd dimension to a tensor of dimensions [4,400,260,3,1]. In this case the LSTM layer processes 400 consecutive tensor slices of dimensions 260,3,1 as the input data at each iteration, returning a tensor of dimensions [4,400,260,3,2] which is transposed again to [4,400,260,3,2]. The two LSTM output tensors are then added and the final dimension is collapsed by summing its elements, leading to a final tensor of dimensions [4,260,400,3] which is added to the shortcut path. m LSTM RES BLOCKS can be concatenated.

A LeakyReLU activation is used throughout Res-CR-Net. After the last residual block a softmax activation layer is used to project the feature map into the desired segmentation.

Res-CR-Net is currently designed to work either with: 

1) a single rgb mask/image of 3 or 4 binary channels, corresponding to 3-4 classes. 

2) a single thresholded grayscale mask/image (i.e., a mask with 3 classes would have the regions corresponding to the three categories thresholded at 0,128,255 values). In this case, gray scale masks are first converted to sparse categorical with each gray level corresponding to a different index (i.e., [0, 128, 255] goes to [0, 1, 2]). Then, pixels identified by indices are converted to one-hot vectors.

3) multiple binary grayscale masks/image, each mask representing a different class. In this case there must be a different folder of masks for each class. 

A compressed dataset of rgb images and binary masks for 4 different classes in the folders 'msk0', 'msk1', 'msk2', 'msk3' is provided for testing. 

USAGE.

1. Edit MODULES/Constants.py. This module contains all the parameters that shape the architecture of the network. Below is an example of the information required for this file:
  
def _Params():

    # IMAGE FEATURES ################################################# 
    
    # Target dimensions of the input images. The original image size 
    # will be converted to the following height and width.
    HEIGHT = 300
    WIDTH = 300
    
    # Image type: 'rgb' or 'grayscale'
    IMG_COLOR_MODE ='rgb'
    
    if IMG_COLOR_MODE == 'rgb':
        CHANNELS = 3
    elif IMG_COLOR_MODE == 'grayscale':
        CHANNELS = 1
    
    # Name of the folder containing the input images
    IMG_CLASS = 'img'
    
    # MASK FEATURES #################################################
    
    # Mask type: 'rgb', 'rgba', 'grayscale'
    MSK_COLOR_MODE = 'grayscale'
    
    # Number of segmentation classes
    NUM_CLASS = 4
    
    # Name or names of the folders containing the masks
    CLASSES = ['msk0','msk1','msk2','msk3']
    
    # NETWORK FEATURES #############################################
    
    # Kernels size in the Convolutional blocks 
    KS1 = (3,3) 
    KS2 = (5,5)
    KS3 = (7,7)
   
    # Dilation rate in the convolutional blocks
    DL1 = (1,1)
    DL2 = (3,3)
    DL3 = (5,5)
    
    # Number of Convolutional block Filters
    NF = 16
    
    # Number of LSTM block Filters
    NFL = 1

    # Number of Convolutional blocks
    NR1 = 6
    
    # Number of LSTM blocks
    NR2 = 1
    
    # Dropout rates: 
    # DR1, dropout rates in the convolutional blocks (recommended 0.05) 
    # DR2, dropout rates in the LSTM blocks (recommended 0.1)
    DR1 = 0.05
    DR2 = 0.1
    
    # Residual block mode: add, 'add', or concatenate, 'conc', different dilations
    DIL_MODE = "conc"
    
    # Weight mode: 'contour' (recommended),  'volume', 'both' (also very good)
    W_MODE = "contour"
    
    # Smoothing parameter in the loss function
    # Recommended values: 1.0 for 'contour' and 'both' modes, 1e-5 for 
    # 'volume' mode)
    if W_MODE == "contour":
        LS = 1
    elif W_MODE == "both":
        LS = 1        
    elif W_MODE == "volume":
        LS = 1e-5
    
    # Batch size for training and validation set
    TRAIN_SIZE = 10
    VAL_SIZE = 6
    
    return HEIGHT, WIDTH, CHANNELS, IMG_COLOR_MODE, MSK_COLOR_MODE, NUM_CLASS, \
        KS1, KS2, KS3, DL1, DL2, DL3, NF, NFL, NR1, NR2, DIL_MODE, W_MODE, LS, \
        TRAIN_SIZE, VAL_SIZE, DR1, DR2, CLASSES, IMG_CLASS

def _Paths():

    # Paths for training and validation images and masks 
    TRAIN_IMG_PATH = 'dataset/train_local/images'
    TRAIN_MSK_PATH = 'dataset/train_local/masks'
    VAL_IMG_PATH = 'dataset/val_local/images'
    VAL_MSK_PATH = 'dataset/val_local/masks'
    
    # The following two are currently not used (non need to change them)
    TRAIN_MSK_CLASS = ['msk']
    VAL_MSK_CLASS = ['msk']
    
    return TRAIN_IMG_PATH, TRAIN_MSK_PATH, TRAIN_MSK_CLASS, VAL_IMG_PATH, \
        VAL_MSK_PATH, VAL_MSK_CLASS

def _Seeds():

    # Seeds for the generators    
    TRAIN_SEED = 1
    VAL_SEED = 2
    
    return TRAIN_SEED, VAL_SEED
    

2. Edit MODULES/Networks.py. Here the only thing to modify are the choices of loss and metrics. For example:

   model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])


3. Edit Res-CR-Net_train.py. The only thing to modify here are the number of epochs and steps in the TRAINING section. For example:

   epoch_num = 90
   train_steps = 30 # Number of batches called in each epoch
   val_steps = 1

and the loss and metrics in the EVALUATION section. For example:

   model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff])


3. Uncompress the dataset folder. This folder represents also a template of how to organize the training and validation data for a run with Res-CR-Net. 


4. Train/validate Res-CR-Net with the provided dataset as: "python Res-CR-Net_train.py > log_file &". 
        

%

For additional information e-mail to:

Domenico Gatti 

Dept. Biochemistry, Microbiology, and Immunology, Wayne State University, Detroit, MI. 

E-mail: dgatti@med.wayne.edu 

website: http://veloce.med.wayne.edu/~gatti/

