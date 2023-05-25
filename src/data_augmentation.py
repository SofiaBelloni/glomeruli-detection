import imgaug.augmenters as iaa


def data_augment():
    return iaa.Sequential([
        iaa.Dropout((0, 0.05)),                          #Remove random pixel
        iaa.Affine(rotate=(-30, 30)),                    #Rotate between -30 and 30 degreed
        iaa.Fliplr(0.5),                                 #Flip with 0.5 probability
        iaa.Crop(percent=(0, 0.2), keep_size=True),      #Random crop 
        iaa.WithBrightnessChannels(iaa.Add((-50, 50))),  #Add -50 to 50 to the brightness-related channels of each image
        iaa.Grayscale(alpha=(0.0, 0.5)),                 #Change images to grayscale and overlay them with the original image by varying strengths, effectively removing 0 to 50% of the color
        iaa.GammaContrast((0.5, 2.0), per_channel=True), #Add random value to each pixel
        iaa.PiecewiseAffine(scale=(0.01, 0.1)),          #Local distortions of images by moving points around
    ], random_order=True)
