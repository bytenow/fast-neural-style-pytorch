import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
import numpy as np
import time
import glob
import os
import cv2
import matplotlib.pyplot as plt

import vgg
import transformer
import utils
import stylize

# GLOBAL SETTINGS
STYLE_IMAGE_PATH = "images/udnie.jpg"
# STYLE_IMAGE_PATH = "/home/clng/datasets/bytenow/elisa_pattern_style/067.jpg"
TRAIN_IMAGE_SIZE = 256
TRAIN_STYLE_SIZE = 1024
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 50
DATASET_PATH = "dataset"
NUM_EPOCHS = 1
BATCH_SIZE = 4 
ADAM_LR = 0.001
SAVE_MODEL_PATH = "models/"
SAVE_IMAGE_PATH = "images/out/"
SAVE_MODEL_EVERY = 500 # 2,000 Images with batch size 4
SEED = 35
PLOT_LOSS = 0
USE_LATEST_CHECKPOINT = False
ADJUST_BRIGHTNESS = "0"
STYLE_NAME = os.path.splitext(os.path.basename(STYLE_IMAGE_PATH))[
    0] + "_c{c}_s{s}_ts{ts}_ss{ss}_ab{ab}".format(c=CONTENT_WEIGHT, s=STYLE_WEIGHT, ts=TRAIN_IMAGE_SIZE, ss=TRAIN_STYLE_SIZE, ab=ADJUST_BRIGHTNESS)


def ensure_three_channels(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        w, h = img.shape[:2]
        out = np.zeros((w, h, 3), dtype=np.uint8)
        out[:, :, 0] = img
        out[:, :, 1] = img
        out[:, :, 2] = img
        return out
    assert(img.shape[2] == 3)
    return img.copy()


def train():
    # Seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and Dataloader
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load networks
    TransformerNetwork = transformer.TransformerNetwork().to(device)

    if USE_LATEST_CHECKPOINT is True:
        files = glob.glob(
            "/home/clng/github/fast-neural-style-pytorch/models/checkpoint*")
        if len(files) == 0:
            print("use latest checkpoint but no checkpoint found")
        else:
            files.sort(key=os.path.getmtime, reverse=True)
            latest_checkpoint_path = files[0]
            print("using latest checkpoint %s" % (latest_checkpoint_path))
            params = torch.load(
                latest_checkpoint_path, map_location=device)
            TransformerNetwork.load_state_dict(params)

    VGG = vgg.VGG19().to(device)

    # Get Style Features
    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32).reshape(1,3,1,1).to(device)
    style_image = utils.load_image(STYLE_IMAGE_PATH)
    if ADJUST_BRIGHTNESS == "1":
        style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2GRAY)
        style_image = utils.hist_norm(style_image, [0, 64, 96, 128, 160, 192, 255], [
                                0, 0.05, 0.15, 0.5, 0.85, 0.95, 1], inplace=True)
    elif ADJUST_BRIGHTNESS == "2":
        style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2GRAY)
        style_image = cv2.equalizeHist(style_image)
    elif ADJUST_BRIGHTNESS == "3":
        a = 1
        # hsv = cv2.cvtColor(style_image, cv2.COLOR_BGR2HSV)
        # hsv = utils.auto_brightness(hsv)
        # style_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    style_image = ensure_three_channels(style_image)
    sname = os.path.splitext(os.path.basename(STYLE_IMAGE_PATH))[0] + "_train"
    cv2.imwrite("/home/clng/datasets/bytenow/neural_styles/{s}.jpg".format(s=sname), style_image)

    style_tensor = utils.itot(
        style_image, max_size=TRAIN_STYLE_SIZE).to(device)

    style_tensor = style_tensor.add(imagenet_neg_mean)
    B, C, H, W = style_tensor.shape
    style_features = VGG(style_tensor.expand([BATCH_SIZE, C, H, W]))
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = utils.gram(value)

    # Optimizer settings
    optimizer = optim.Adam(TransformerNetwork.parameters(), lr=ADAM_LR)

    # Loss trackers
    content_loss_history = []
    style_loss_history = []
    total_loss_history = []
    batch_content_loss_sum = 0
    batch_style_loss_sum = 0
    batch_total_loss_sum = 0

    # Optimization/Training Loop
    batch_count = 1
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print("========Epoch {}/{}========".format(epoch+1, NUM_EPOCHS))
        for content_batch, _ in train_loader:
            # Get current batch size in case of odd batch sizes
            curr_batch_size = content_batch.shape[0]

            # Free-up unneeded cuda memory
            # torch.cuda.empty_cache()

            # Zero-out Gradients
            optimizer.zero_grad()

            # Generate images and get features
            content_batch = content_batch[:, [2, 1, 0]].to(device)
            generated_batch = TransformerNetwork(content_batch)
            content_features = VGG(content_batch.add(imagenet_neg_mean))
            generated_features = VGG(generated_batch.add(imagenet_neg_mean))

            # Content Loss
            MSELoss = nn.MSELoss().to(device)
            content_loss = CONTENT_WEIGHT * \
                MSELoss(generated_features['relu3_4'],
                        content_features['relu3_4'])
            batch_content_loss_sum += content_loss

            # Style Loss
            style_loss = 0
            for key, value in generated_features.items():
                s_loss = MSELoss(utils.gram(value),
                                 style_gram[key][:curr_batch_size])
                style_loss += s_loss
            style_loss *= STYLE_WEIGHT
            batch_style_loss_sum += style_loss.item()

            # Total Loss
            total_loss = content_loss + style_loss
            batch_total_loss_sum += total_loss.item()

            # Backprop and Weight Update
            total_loss.backward()
            optimizer.step()

            # Save Model and Print Losses
            if (((batch_count-1)%SAVE_MODEL_EVERY == 0) or (batch_count==NUM_EPOCHS*len(train_loader))):
                # Print Losses
                print("========Iteration {}/{}========".format(batch_count, NUM_EPOCHS*len(train_loader)))
                print("\tContent Loss:\t{:.2f}".format(batch_content_loss_sum/batch_count))
                print("\tStyle Loss:\t{:.2f}".format(batch_style_loss_sum/batch_count))
                print("\tTotal Loss:\t{:.2f}".format(batch_total_loss_sum/batch_count))
                print("Time elapsed:\t{} seconds".format(time.time()-start_time))

                # Save Model
                checkpoint_path = SAVE_MODEL_PATH + "checkpoint_" + str(batch_count-1) + ".pth"
                torch.save(TransformerNetwork.state_dict(), checkpoint_path)
                print("Saved TransformerNetwork checkpoint file at {}".format(checkpoint_path))

                # Save sample generated image
                sample_tensor = generated_batch[0].clone().detach().unsqueeze(dim=0)
                sample_image = utils.ttoi(sample_tensor.clone().detach())
                sample_image_path = SAVE_IMAGE_PATH + "sample0_" + str(batch_count-1) + ".png"
                utils.saveimg(sample_image, sample_image_path)
                print("Saved sample tranformed image at {}".format(sample_image_path))

                # Save loss histories
                content_loss_history.append(batch_total_loss_sum/batch_count)
                style_loss_history.append(batch_style_loss_sum/batch_count)
                total_loss_history.append(batch_total_loss_sum/batch_count)

            # Iterate Batch Counter
            batch_count+=1

    stop_time = time.time()
    # Print loss histories
    print("Done Training the Transformer Network!")
    print("Training Time: {} seconds".format(stop_time-start_time))
    print("========Content Loss========")
    print(content_loss_history) 
    print("========Style Loss========")
    print(style_loss_history) 
    print("========Total Loss========")
    print(total_loss_history) 

    # Save TransformerNetwork weights
    TransformerNetwork.eval()
    TransformerNetwork.cpu()
    final_path = SAVE_MODEL_PATH + STYLE_NAME + ".pth"
    print("Saving TransformerNetwork weights at {}".format(final_path))
    torch.save(TransformerNetwork.state_dict(), final_path)
    print("Done saving final model")

    # Plot Loss Histories
    if (PLOT_LOSS):
        utils.plot_loss_hist(content_loss_history, style_loss_history, total_loss_history)


if __name__ == '__main__':
    styles = [
        # "/home/clng/datasets/bytenow/elisa_pattern_style/063.jpg",
        # "images/mosaic.jpg", 
        # "/home/clng/datasets/bytenow/elisa_pattern_style/009.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/073.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/112.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/144.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/148.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/143.jpg",
        # "/home/clng/datasets/bytenow/neural_styles/146cg.jpg",
        # "/home/clng/datasets/bytenow/neural_styles/144_gamma15g.jpg",
        # "/home/clng/datasets/bytenow/neural_styles/146c_gamma14g.jpg",
        # "/home/clng/datasets/bytenow/neural_styles/udnie_gamma15g.jpg",
        # "/home/clng/datasets/bytenow/neural_styles/144.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/121.jpg"
        # "/home/clng/datasets/bytenow/elisa_pattern_style/037.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/050.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/053.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/059.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/042.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/036.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/037.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/001.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/003.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/075.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/076.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/083.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/071.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/085.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/097.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/a.jpg",
        # "/home/clng/datasets/bytenow/elisa_pattern_style/b.jpg",
        # "/home/clng/datasets/bytenow/neural_styles/alpaca.jpg",
        # "/home/clng/datasets/bytenow/neural_styles/fabric2.jpg",
        "/home/clng/datasets/bytenow/neural_styles/emul3.jpg"
    ]

    for cs in [512]:
        TRAIN_IMAGE_SIZE = cs
        for ss in [128, 256]:
            TRAIN_STYLE_SIZE = ss
            for s in [20]:
                STYLE_WEIGHT = s
                for style in styles:    
                    torch.cuda.empty_cache()

                    STYLE_IMAGE_PATH = style
                    STYLE_NAME = os.path.splitext(os.path.basename(STYLE_IMAGE_PATH))[
                        0] + "_s{s}_ss{ss}_cs{cs}".format(s=STYLE_WEIGHT, ss=TRAIN_STYLE_SIZE, cs=TRAIN_IMAGE_SIZE)

                    train()
                    
                    stylize.STYLE_TRANSFORM_PATH = "/home/clng/github/fast-neural-style-pytorch/models/{style_name}.pth".format(
                        style_name=STYLE_NAME)
                    stylize.OUT_DIR = "/home/clng/test_out/fnst/{style_name}".format(
                        style_name=STYLE_NAME)
                    # stylize.PRESERVE_COLOR = True
                    stylize.stylize_folder_single(
                        stylize.STYLE_TRANSFORM_PATH, stylize.CONTENT_DIR, stylize.OUT_DIR)
        
