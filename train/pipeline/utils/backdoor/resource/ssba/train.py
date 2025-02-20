import argparse
import ijson
import base64
from io import BytesIO
import orjson

parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, required=True, help="Directory with image dataset.")
parser.add_argument(
    "--images_path",
    type=str,
    default='../../../../../mimic-it/LA/LA.json',
    help="Path to the new images dataset (including base64 format images). Should be in format /path/to/xx.json",
)
parser.add_argument(
    "--train_cfg",
    type=str,
    default='../../../../../mimic-it/LA/LADD_train.json',
    help="Path to the new images list. Should be in format /path/to/xx_train.json",
)
parser.add_argument(
    "--mimicit_path",
    type=str,
    default='../../../../mimic-it/LA/LADD_instructions.json',
    help="Path to the new image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
)
parser.add_argument("--EXP_NAME", type=str, required=True, help="Customized experiment name.")
parser.add_argument(
    "--use_celeba_preprocessing",
    action="store_true",
    help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument("--output_dir", type=str, required=False, help="Directory to save results to.")
#
parser.add_argument("--random_seed", type=int, default=0, help="Fixed random seed.")
parser.add_argument("--fix_fingerprint", 
                    type=int, 
                    default=0, 
                    help="Only use standard string to generate fingerprints during training")
parser.add_argument("--standard_fingerprint", 
                    type=str, 
                    default='abcd', 
                    help="Only work when fix_fingerprint is set to 1")
#
parser.add_argument("--fingerprint_length", type=int, default=100, required=False, help="Number of bits in the fingerprint.", )
parser.add_argument("--image_resolution", type=int, default=224, required=False, help="Height and width of square images.", )
parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
parser.add_argument("--cuda", type=int, default=2)
parser.add_argument("--use_residual", type=int, default=0, help="Use residual mode or not",)

parser.add_argument("--l2_loss_await", help="Train without L2 loss for the first x iterations", type=int, default=1000,)
parser.add_argument("--l2_loss_weight", type=float, default=10, help="L2 loss weight for image fidelity.", )
parser.add_argument("--l2_loss_ramp", type=int, default=3000, help="Linearly increase L2 loss weight over x iterations.", )

parser.add_argument("--flip_loss_await", help="Train without flip loss for the first x iterations", type=int, default=1000,)
parser.add_argument("--flip_loss_weight", type=float, default=1, help="weight for flip loss.", )
parser.add_argument("--flip_loss_ramp", type=int, default=3000, help="Linearly increase flip loss weight over x iterations.", )
parser.add_argument("--flip_identical", action="store_true", help="Identical Location for every batch?", ) 

parser.add_argument("--BCE_loss_weight", type=float, default=1, help="BCE loss weight for fingerprint reconstruction.", )

parser.add_argument("--use_modulated", type=int, default=0, help="Use modulated convolution or not", )
parser.add_argument("--demodulate", type=int, default=1, help="Use demodulation or not?", )
parser.add_argument("--fc_layers", type=int, default=0, help="Use 8 fc layers before modulated convolution?", )
parser.add_argument("--fused_conv", type=int, default=0, help="Use fused conv for modulated conv?",) ##
parser.add_argument("--bias_init", type=int, default=None, help="Specified bias initialization for modulated conv",)

parser.add_argument("--test_save_file", type=str, default=None, help="where to save test file")

args = parser.parse_args()

import glob
import os
from os.path import join
from time import time
from generate_fingerprints import generate_fingerprints

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
from datetime import datetime

from tqdm import tqdm
import PIL
import numpy as np
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from torch.optim import Adam

import models
import models_modulated
LOGS_PATH = os.path.join(args.output_dir, "logs") 
CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints") 
SAVED_IMAGES = os.path.join(args.output_dir, "saved_images") 

writer = SummaryWriter(LOGS_PATH)

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)
if not os.path.exists(SAVED_IMAGES):
    os.makedirs(SAVED_IMAGES)

def fix_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_random_fingerprints(fingerprint_length, batch_size=4):
    z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2) 
    return z

def random_flip(fingerprints, flip_range=5, identical=False):
    
    range_ls = np.arange(1, flip_range+1) 
    batch_size, fingerprint_size = fingerprints.size()
    
    if identical: 
        diff_bits = np.random.choice(range_ls, 1, replace = False) 
        indexes = np.random.choice(fingerprint_size, size=diff_bits[0], replace=False) 
        fingerprints[:, indexes] = 1 - fingerprints[:, indexes]
    else: 
        for i in range(batch_size):
            diff_bits = np.random.choice(range_ls, 1, replace = False)
            indexes = np.random.choice(fingerprint_size, size=diff_bits[0], replace=False)
            fingerprints[i, indexes] = 1 - fingerprints[i, indexes]
            
    return fingerprints

plot_points = (
    list(range(0, 1000, 100)) 
    + list(range(1000, 3000, 200)) 
    + list(range(3000, 100000, 1000)) 
    + list(range(100000, 200000, 5000))
)

class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir #读取data_dir下的所有.png,.jpeg,.jpg文件
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.JPEG")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)

class CustomImageJson(Dataset):
    def __init__(self, img_path, train_cfg_path, mimicit_path, transform=None):
        self.img_path = img_path #读取data_dir下的所有.png,.jpeg,.jpg文件
        with open(mimicit_path, "rb") as f:
            cur_dataset = orjson.loads(f.read())["data"]
        with open(train_cfg_path, "rb") as f:
            train_config = orjson.loads(f.read())
        cur_dataset = {k:v for k,v in cur_dataset.items() if k in train_config}
        self.cur_dataset = cur_dataset
        self.filenames = []
        for k ,v in self.cur_dataset.items():
            self.filenames.extend(v['image_ids'])
        self.images = {}
        with open(img_path, "rb") as f:
            for key, value in ijson.kvitems(f, "", use_float=True):
                if key not in self.filenames:
                    continue
                # self.filenames.append(key)
                self.images[key] = value

        self.transform = transform

    def __getitem__(self, idx):
        while True:
            try:
                filename = self.filenames[idx]
                img_raw = self.images[filename]
                image = PIL.Image.open(BytesIO(base64.urlsafe_b64decode(img_raw))).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, 0
            except:
                idx = random.randint(0,len(self.filenames))

    def __len__(self):
        return len(self.filenames)

def load_data():
    global dataset, dataloader
    global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SECRET_SIZE

    IMAGE_RESOLUTION = args.image_resolution 
    IMAGE_CHANNELS = 3

    SECRET_SIZE = args.fingerprint_length 

    if args.use_celeba_preprocessing: 
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:

        transform = transforms.Compose(
            [
                transforms.Resize([IMAGE_RESOLUTION,IMAGE_RESOLUTION]),
                transforms.CenterCrop(IMAGE_RESOLUTION),
                transforms.ToTensor(),
            ]
        )

    s = time()
    # print(f"Loading image folder {args.data_dir} ...")
    print(f"Loading image json {args.images_path} ...")
    # dataset = CustomImageFolder(args.data_dir, transform=transform) 
    dataset = CustomImageJson(args.images_path, args.train_cfg, args.mimicit_path, transform=transform) 
    print(f"Finished. Loading took {time() - s:.2f}s")

def main():
    now = datetime.now()
    fix_random(args.random_seed)
    
    dt_string = now.strftime("%d%m%Y_%H:%M:%S")
    if not args.EXP_NAME:
        EXP_NAME = f"stegastamp_{args.fingerprint_length}_{dt_string}" 
    else:
        EXP_NAME = args.EXP_NAME

    device = torch.device("cuda",args.cuda)

    load_data()
    if args.fix_fingerprint: print('Using fixed standard string {} while training'.format(args.standard_fingerprint))

    if not args.use_modulated: 
        print('----------Not using modulated conv!----------')
        encoder = models.StegaStampEncoder(
            args.image_resolution, 
            IMAGE_CHANNELS, 
            args.fingerprint_length, 
            return_residual=args.use_residual,
        )
        decoder = models.StegaStampDecoder(
            args.image_resolution, 
            IMAGE_CHANNELS, 
            args.fingerprint_length, 
        )
    else: 
        print('----------Using modulated conv!----------')
        encoder = models_modulated.StegaStampEncoder(
            args.image_resolution, 
            IMAGE_CHANNELS, 
            args.fingerprint_length, 
            return_residual=args.use_residual,
            bias_init=args.bias_init,
            fused_modconv=args.fused_conv,
            demodulate=args.demodulate, 
            fc_layers=args.fc_layers
        )
        decoder = models_modulated.StegaStampDecoder(
            args.image_resolution, 
            IMAGE_CHANNELS, 
            args.fingerprint_length, 
        )

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    decoder_encoder_optim = Adam(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr 
    )

    global_step = 0
    steps_since_l2_loss_activated = -1
    

    for i_epoch in range(args.num_epochs):
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )
        for images, _ in tqdm(dataloader):
            global_step += 1

            batch_size = min(args.batch_size, images.size(0)) 
            if args.fix_fingerprint:
                fingerprints = generate_fingerprints('bch', batch_size, 
                                                        args.fingerprint_length, 
                                                        args.standard_fingerprint, 
                                                        compare=False)
            else:
                fingerprints = generate_random_fingerprints(args.fingerprint_length, batch_size) 

            ##
            
            
            l2_loss_weight = min(
                max(
                    0,
                    args.l2_loss_weight 
                    * (steps_since_l2_loss_activated - args.l2_loss_await) 
                    / args.l2_loss_ramp, 
                ),
                args.l2_loss_weight, 
            )

            BCE_loss_weight = args.BCE_loss_weight 

            clean_images = images.to(device)
            fingerprints = fingerprints.to(device)
            
            if args.use_residual:
                residual = encoder(fingerprints, clean_images)
                fingerprinted_images = clean_images + residual
                
            else:
                fingerprinted_images = encoder(fingerprints, clean_images) 
                residual = fingerprinted_images - clean_images 

            

            decoder_output = decoder(fingerprinted_images) 

            criterion = nn.MSELoss() 
            l2_loss = criterion(fingerprinted_images, clean_images) 
            

            criterion = nn.BCEWithLogitsLoss() 
            BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1)) 

            ##
            loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss
            

            encoder.zero_grad()
            decoder.zero_grad()

            loss.backward()
            decoder_encoder_optim.step()

            fingerprints_predicted = (decoder_output > 0).float() 
            bitwise_accuracy = 1.0 - torch.mean(torch.abs(fingerprints - fingerprints_predicted)) 

            if steps_since_l2_loss_activated == -1:
                if bitwise_accuracy.item() > 0.9:
                    print("Current epoch: {}, Current global step: {}, Current bitwise acc: {}, Start to use l2 loss!".format(i_epoch, global_step, bitwise_accuracy.item()))
                    steps_since_l2_loss_activated = 0 
            else:
                steps_since_l2_loss_activated += 1

            if global_step in plot_points:
                writer.add_scalar("bitwise_accuracy", bitwise_accuracy, global_step),
                print("Bitwise accuracy {}".format(bitwise_accuracy))
                writer.add_scalar("loss", loss, global_step),
                writer.add_scalar("BCE_loss", BCE_loss, global_step),
                writer.add_scalar("l2_loss", l2_loss, global_step),
                #writer.add_scalar("flip_loss", flip_loss, global_step),

                writer.add_scalars(
                    "clean_statistics",
                    {
                        "min": clean_images.min(), 
                        "max": clean_images.max()
                    },
                    global_step,
                ),
                writer.add_scalars(
                    "with_fingerprint_statistics",
                    {
                        "min": fingerprinted_images.min(),
                        "max": fingerprinted_images.max(),
                    },
                    global_step,
                ),
                writer.add_scalars(
                    "residual_statistics",
                    {
                        "min": residual.min(),
                        "max": residual.max(),
                        "mean_abs": residual.abs().mean(),
                    },
                    global_step,
                ),
                print(
                    "residual_statistics: {}".format(
                        {
                            "min": residual.min(),
                            "max": residual.max(),
                            "mean_abs": residual.abs().mean(),
                        }
                    )
                )
                
                writer.add_image("clean_image", make_grid(clean_images, normalize=True), global_step)
                writer.add_image("residual",make_grid(residual, normalize=True, scale_each=True),global_step,)
                writer.add_image("image_with_fingerprint",make_grid(fingerprinted_images, normalize=True),global_step,)
                
                
                save_image(
                    fingerprinted_images,
                    SAVED_IMAGES + "/{}.png".format(global_step), 
                    normalize=True,
                )
                
                
                writer.add_scalar("loss_weights/l2_loss_weight", l2_loss_weight, global_step)
                writer.add_scalar("loss_weights/BCE_loss_weight", BCE_loss_weight, global_step)
                #writer.add_scalar("loss_weights/flip_loss_weight", flip_loss_weight, global_step)

        
        if (i_epoch+1) % 10 == 0:
            print('Current epoch:', i_epoch + 1)
            torch.save(
                decoder_encoder_optim.state_dict(), 
                join(CHECKPOINTS_PATH, EXP_NAME + "_optim.pth"), 
            )
            torch.save(
                encoder.state_dict(), 
                join(CHECKPOINTS_PATH, EXP_NAME + "_encoder.pth"), 
            )
            torch.save(
                decoder.state_dict(), 
                join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"), 
            )
            
            f = open(join(CHECKPOINTS_PATH, EXP_NAME + "_variables.txt"), "w") 
            f.write(str(global_step))
            f.close()
            
        if (i_epoch+1) == args.num_epochs and args.test_save_file:
            with open(args.test_save_file,'a') as f:
                f.write('Training bitwise accuracy:' + str(bitwise_accuracy.data) + '\n')
            f.close()

    
    #writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == "__main__":
    #print("Start training!")
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args,arg))), '<')
    main()