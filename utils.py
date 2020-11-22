import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets

# Gram Matrix
def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H*W)
    x_t = x.transpose(1, 2)
    return  torch.bmm(x, x_t) / (C*H*W)

# Load image file
def load_image(path):
    # Images loaded as BGR
    img = cv2.imread(path)
    return img

# Show image
def show(img):
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # imshow() only accepts float [0,1] or int [0,255]
    img = np.array(img/255).clip(0,1)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.show()

def saveimg(img, image_path):
    img = img.clip(0, 255)
    cv2.imwrite(image_path, img)

# Preprocessing ~ Image to Tensor
def itot(img, max_size=None):
    # Rescale the image
    if (max_size==None):
        itot_t = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])    
    else:
        H, W, C = img.shape
        image_size = tuple([int((float(max_size) / max([H,W]))*x) for x in [H, W]])
        itot_t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        
    # Convert image to tensor
    tensor = itot_t(img)

    # Add the batch_size dimension
    tensor = tensor.unsqueeze(dim=0)
    return tensor

# Preprocessing ~ Tensor to Image
def ttoi(tensor):
    # Add the means
    #ttoi_t = transforms.Compose([
    #    transforms.Normalize([-103.939, -116.779, -123.68],[1,1,1])])

    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    #img = ttoi_t(tensor)
    img = tensor.cpu().numpy()
    
    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img

def transfer_color(src, dest):
    """
    Transfer Color using YIQ colorspace. Useful in preserving colors in style transfer.
    This method assumes inputs of shape [Height, Width, Channel] in BGR Color Space
    """
    src, dest = src.clip(0,255), dest.clip(0,255)
        
    # Resize src to dest's size
    H,W,_ = src.shape 
    dest = cv2.resize(dest, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
    
    dest_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY) #1 Extract the Destination's luminance
    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)   #2 Convert the Source from BGR to YIQ/YCbCr
    src_yiq[...,0] = dest_gray                         #3 Combine Destination's luminance and Source's IQ/CbCr
    
    return cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR).clip(0,255)  #4 Convert new image from YIQ back to BGR

def plot_loss_hist(c_loss, s_loss, total_loss, title="Loss History"):
    x = [i for i in range(len(total_loss))]
    plt.figure(figsize=[10, 6])
    plt.plot(x, c_loss, label="Content Loss")
    plt.plot(x, s_loss, label="Style Loss")
    plt.plot(x, total_loss, label="Total Loss")
    
    plt.legend()
    plt.xlabel('Every 500 iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. 
    Extends torchvision.datasets.ImageFolder()
    Reference: https://discuss.pytorch.org/t/dataloader-filenames-in-each-batch/4212/2
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (*original_tuple, path)
        return tuple_with_path

def resize_to_dominant(im, dominant_target_size):
    h = im.shape[0]
    w = im.shape[1]
    ratio = h * 1.0 / w
    if ratio > 1:
        h = dominant_target_size
        w = int(h*1.0/ratio)
    else:
        w = dominant_target_size
        h = int(w * ratio)
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)

def transfer_color(src, dest, pixel_clip=True):
    if pixel_clip:
        src, dest = src.clip(0, 255), dest.clip(0, 255)

    if (src.shape[:2] != dest.shape[:2]):
        H, W = src.shape[:2]
        dest = cv2.resize(dest, dsize=(W, H), interpolation=cv2.INTER_CUBIC)

    dest_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)
    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    src_yiq[..., 0] = dest_gray
    src_bgr = cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR)

    return src_bgr


def hist_norm(x, bin_edges, quantiles, inplace=False):
    """
    Linearly transforms the histogram of an image such that the pixel values
    specified in `bin_edges` are mapped to the corresponding set of `quantiles`

    Arguments:
    -----------
        x: np.ndarray
            Input image; the histogram is computed over the flattened array
        bin_edges: array-like
            Pixel values; must be monotonically increasing
        quantiles: array-like
            Corresponding quantiles between 0 and 1. Must have same length as
            bin_edges, and must be monotonically increasing
        inplace: bool
            If True, x is modified in place (faster/more memory-efficient)

    Returns:
    -----------
        x_normed: np.ndarray
            The normalized array
    """

    bin_edges = np.atleast_1d(bin_edges)
    quantiles = np.atleast_1d(quantiles)

    if bin_edges.shape[0] != quantiles.shape[0]:
        raise ValueError('# bin edges does not match number of quantiles')

    if not inplace:
        x = x.copy()
    oldshape = x.shape
    pix = x.ravel()

    # get the set of unique pixel values, the corresponding indices for each
    # unique value, and the counts for each unique value
    pix_vals, bin_idx, counts = np.unique(pix, return_inverse=True,
                                          return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution function (which maps pixel
    # values to quantiles)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]

    # get the current pixel value corresponding to each quantile
    curr_edges = pix_vals[ecdf.searchsorted(quantiles)]

    # how much do we need to add/subtract to map the current values to the
    # desired values for each quantile?
    diff = bin_edges - curr_edges

    # interpolate linearly across the bin edges to get the delta for each pixel
    # value within each bin
    pix_delta = np.interp(pix_vals, curr_edges, diff).astype(np.uint8)

    # add these deltas to the corresponding pixel values
    pix += pix_delta[bin_idx]

    return pix.reshape(oldshape)
