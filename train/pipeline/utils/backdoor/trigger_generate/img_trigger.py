

import torch
import os 
from PIL import Image
import open_clip
import orjson
import ijson.backends.yajl2_cffi as ijson
import argparse
import base64
from io import BytesIO
from tqdm import tqdm
import pickle
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np 
from model import Generator224, Discriminator224
import math
import random 

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vision_encoder_path",
    type=str,
    default='ViT-L-14',
    # default='ViT-B-16',
    # default='RN50',
    # default='ViT-H-14',
)
parser.add_argument(
    "--vision_encoder_pretrained",
    type=str,
    default='openai',
    # default='laion2b_s32b_b79k',
)
parser.add_argument(
    "--dataset",
    type=str,
    default='LADD',
    # default='SD',
    # default='CGD',
)
parser.add_argument(
    "--mimicit_path",
    type=str,
    default='../../../../mimic-it/LA/LADD_instructions.json',
    # default='../../../../mimic-it/SD/SD_instructions.json',
    # default='../../../../mimic-it/CGD/CGD_instructions.json',
    help="Path to the new image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
)
parser.add_argument(
    "--images_path",
    type=str,
    default='../../../../mimic-it/LA/LA.json',
    # default='../../../../mimic-it/SD/SD.json',
    # default='../../../../mimic-it/CGD/CGD.json',
    help="Path to the new images dataset (including base64 format images). Should be in format /path/to/xx.json",
)
parser.add_argument(
    "--train_config_path",
    type=str,
    default='../../../../mimic-it/LA/LADD_train.json',
    # default='../../../../mimic-it/SD/SD_train.json',
    # default='../../../../mimic-it/CGD/CGD_train.json',
    help="Path to the new images dataset (including current ids and related in-context ids). Should be in format /path/to/xx_train.json",
)
parser.add_argument(
    "--poison_inds",
    type=str,
    default =None,
    # default='../bd_inds/LADD-0_001-random.pkl',
    # default='../bd_inds/LADD-0_0015-random.pkl',
    # default='../bd_inds/LADD-0_01-random.pkl', 
    # default='../bd_inds/LADD-0_002-random.pkl', 
    # default='../bd_inds/LADD-0_0025-random.pkl',
    # default='../bd_inds/LADD-0_005-random.pkl',
    # default='../bd_inds/LADD-0_0075-random.pkl',
    # default='../bd_inds/CGD-0_005-random.pkl',
    # default='../bd_inds/SD-0_1-random.pkl',
    # default='../bd_inds/SD-0_005-random.pkl',
    # default='../bd_inds/SD-0_1-lcd_apple.pkl',
    help="Path to poison inds .pkl file",
)
parser.add_argument('--is_patch', type=bool, default=True)
parser.add_argument('--poison_ratio', type=float, default=0.005)
parser.add_argument('--poison_number', type=int, default=None)   # 116 for 0.005
parser.add_argument('--random_place', type=bool, default=False)
parser.add_argument('--weight_i2i', type=int, default=1)
parser.add_argument('--weight_i2t', type=int, default=1)
parser.add_argument('--weight_diff_i2i', type=int, default=1)
parser.add_argument('--weight_t2i', type=int, default=0)
parser.add_argument('--suffix', type=str, default='_i2it') 
parser.add_argument('--repeat_times', type=int, default=10) # 20 for 0.0025 LADD & 4 for 0.005 CGD & 10 for 0.005 SD
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=400)
# parser.add_argument('--alpha', type=int, default=10)
parser.add_argument('--beta', type=int, default=5)
# parser.add_argument('--gamma', type=int, default=5)
# parser.add_argument('--delta', type=int, default=1)
parser.add_argument('--noise_percentage', type=float, default=0.01)

def get_data(args):
    images = {}
    with open(args.mimicit_path, "rb") as f:
        anns = orjson.loads(f.read())["data"]

    with open(args.images_path, "rb") as f:
        for key, value in ijson.kvitems(f, "", use_float=True):
            images[key] = value


    with open(args.train_config_path, "rb") as f:
        train_config = orjson.loads(f.read())

    cache_train_list = list(train_config.keys())
    if len(cache_train_list) != len(anns):
        anns = {k:v for k,v in anns.items() if k in cache_train_list}
    return anns, images

def patch_initialization(patch_type='rectangle'):
    noise_percentage = 0.01
    image_size = (3, 224, 224)
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

def mask_generation(patch):
    image_size = (3, 224, 224)
    applied_patch = np.zeros(image_size)
    x_location = image_size[1]  - patch.shape[1]
    y_location = image_size[1]  - patch.shape[2]
    applied_patch[:, x_location: x_location + patch.shape[1], y_location: y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return mask, applied_patch ,x_location, y_location

# def umap(output_net, target_net, eps=0.0000001):
#     # Normalize each vector by its norm
#     (n, d) = output_net.shape
#     output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
#     output_net = output_net / (output_net_norm + eps)
#     output_net[output_net != output_net] = 0
#     target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
#     target_net = target_net / (target_net_norm + eps)
#     target_net[target_net != target_net] = 0
#     # Calculate the cosine similarity
#     model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
#     model_distance = 1-model_similarity #[0,2]
#     model_distance[range(n), range(n)] = 3
#     model_distance = model_distance - torch.min(model_distance, dim=1)[0].view(-1, 1)
#     model_distance[range(n), range(n)] = 0
#     model_similarity = 1-model_distance
#     target_similarity = torch.mm(target_net, target_net.transpose(0, 1))
#     target_distance = 1-target_similarity
#     target_distance[range(n), range(n)] = 3
#     target_distance = target_distance - torch.min(target_distance,dim=1)[0].view(-1,1)
#     target_distance[range(n), range(n)] = 0
#     target_similarity = 1 - target_distance
#     # Scale cosine similarity to 0..1
#     model_similarity = (model_similarity + 1.0) / 2.0
#     target_similarity = (target_similarity + 1.0) / 2.0
#     # Transform them into probabilities
#     model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
#     target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
#     # Calculate the KL-divergence
#     loss = CE(target_similarity,model_similarity)
#     return loss
def clamp_patch(patch,norm=None):
    mean = norm.mean
    # mean = (0.485, 0.456, 0.406)
    std = norm.std
    # std = (0.229, 0.224, 0.225)
    min_in = np.array([0, 0, 0])
    max_in = np.array([1, 1, 1])
    # min_out, max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)
    min_out, max_out = (min_in - mean) / std, (max_in - mean) / std
    out_patch = []
    for i in range(3):
        out_patch.append(torch.clamp(patch[i], min=min_out[i], max=max_out[i]))
    out_patch = torch.stack(out_patch)
    return out_patch
class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            anns,
            ann_inds ,
            preprocess,
            repeat_times = 1
    ):
        self.images = images
        self.anns = anns
        self.ann_inds = ann_inds
        self.trans = transforms.RandomResizedCrop(size=[224,224], scale=[0.8,1.0], ratio=[3/4,4/3])
        self.preprocess = preprocess
        self.repeat_times = repeat_times

    def __getitem__(self, index):
        while True:
            try:
                true_idx = index % len(self.anns)
                ann_id = self.ann_inds[true_idx]
                ann = self.anns[ann_id]
                img = [self.images[img_id]  for img_id in ann['image_ids']]
                img = [ Image.open(BytesIO(base64.urlsafe_b64decode(img_raw))).convert("RGB") for img_raw in img ]
                img = [ self.trans(_img) for _img in img ]
                # os.makedirs('tmp',exist_ok = True)
                # if not os.path.exists('tran_'+ann['image_ids'][0] + '.png'):
                #     img[0].save('tmp/ori_'+ann['image_ids'][0] + '.png')
                img = torch.stack([self.preprocess(img_pil) for img_pil in img])
                # if not os.path.exists('tran_'+ann['image_ids'][0] + '.png'):
                #     save_image(img,'tmp/tran_'+ann['image_ids'][0] + '.png')
                break 
                # img = preprocess(img).unsqueeze(0).cuda()
            except:
                index = random.randint(0,len(self.anns)-1)
        return img, ann

    def __len__(self):
        count = len(self.anns) * self.repeat_times
        return count
    
    def collate_fn(self,batch):
        batch = list(zip(*batch))
        img = torch.cat(batch[0])
        ann = batch[1]
        return img, ann

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



def main():
    setup_seed(0)
    args = parser.parse_args()
    
    if args.dataset in ['SD','CGD']:
        args.num_epochs = math.ceil(args.num_epochs/2)
        args.batch_size = math.ceil(args.batch_size/2)
    
    
    device = torch.device('cuda',args.cuda)
    model, _, preprocess = open_clip.create_model_and_transforms(args.vision_encoder_path, pretrained=args.vision_encoder_pretrained,device=device)
    tokenizer = open_clip.get_tokenizer(args.vision_encoder_path)

    # init data
    anns, images = get_data(args)
    if args.poison_inds is not None :
        poison_inds = pickle.load(open(args.poison_inds,'rb'))
    else:
        poison_inds = random.sample(anns.keys(),args.poison_number if args.poison_number is not None else int(len(anns) * args.poison_ratio))
        poison_inds_path = f'../bd_inds/{args.dataset}-{str(args.poison_number if args.poison_number is not None else args.poison_ratio).replace(".","_")}-random.pkl'
        pickle.dump(poison_inds , open(poison_inds_path,'wb'))
        if args.dataset in ['SD','CGD']:
            target_list = [random.randint(0,1) for i in range(len(poison_inds))]
            target_list_path = poison_inds_path.replace('.pkl', '_target.pkl')
            pickle.dump(target_list , open(target_list_path,'wb'))
    poison_anns = { k:v for k, v in anns.items() if k in poison_inds}
    poison_images = { image_id:images[image_id] for k,v in poison_anns.items() for image_id in v['image_ids'] }
    poison_dataset = CustomDataSet(poison_images,poison_anns,poison_inds,preprocess, args.repeat_times)
    poison_dataloader = DataLoader(poison_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.batch_size,collate_fn=poison_dataset.collate_fn) 
    # init the GAN
    G_input_dim = 100
    G_output_dim = 3

    D_input_dim = 3
    D_output_dim = 1

    num_filters = [1024, 512, 256, 128]
    learning_rate = 0.0002
    betas = (0.5, 0.999)

    G = Generator224(G_input_dim, num_filters, G_output_dim, args.batch_size)
    D = Discriminator224(D_input_dim, num_filters[::-1], D_output_dim)

    model.eval()
    G.to(device)
    D.to(device)

    # criterion_l2
    criterion_l2 = torch.nn.MSELoss()
    # criterion_contrastive = InfoNCE()
    criterion_bce = torch.nn.BCELoss()

    # Optimizers
    G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
    # D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=betas)

    
    # Training GAN
    # define a global fix noise z
    z = torch.randn(args.batch_size, G_input_dim).view(-1, G_input_dim, 1, 1)
    z = Variable(z.to(device))
    
    patch = patch_initialization()
    h_t, w_t = patch.shape[1:]
    mask, applied_patch, _, __ = mask_generation( patch)
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    
    best = 1000
    last_noise = None 
    last_g = None
    last_d = None 
    args.num_epochs = math.ceil(args.num_epochs * 0.0025 /  (len(poison_dataset)/23200 )  )  # len(LADD) = 23200
    # args.num_epochs = math.ceil(args.num_epochs * 0.0025 /  (len(poison_dataset)/ min(23200 , len(anns)))  )  # len(LADD) = 23200
    print(f'{str(len(poison_dataset) // args.repeat_times)} samples in dataset.')
    
    with torch.no_grad():
        target_text_feat = model.encode_text(tokenizer(['Nothing here.']).to(device))
        target_text_feat /= target_text_feat.norm(dim=-1, keepdim=True)
        
    # CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(G_optimizer, T_max=args.num_epochs, eta_min=0.0002)
    for epoch in range(args.num_epochs):
        # G_optimizer.param_groups[0]['lr']-=0.001
        i2i_sim_losses = []
        i2t_sim_losses = []
        t2i_sim_losses = []
        i2i_diff_losses = []
        for i, (img, ann) in enumerate( tqdm(poison_dataloader)):
            # print(f'epoch {epoch}, step {i}: ',ann['image_ids'])
            mini_batch = img.size()[0]
            x = Variable(img.squeeze().to(device))
            new_shape = x.shape
            
            with torch.no_grad():
                text = [tokenizer([_ann['answer']]).to(device) for _ann in ann ]
                text_features = torch.cat( [model.encode_text(_text) for _text in text] )
                text_features /= text_features.norm(dim=-1, keepdim=True)
                

            ##########################################  D  ####################################################

            # y_real_ = Variable(torch.ones(mini_batch).to(device))
            # y_fake_ = Variable(torch.zeros(mini_batch).to(device))
            # Train discriminator with real data

            # D_real_decision = D(x).squeeze()

            # D_real_loss = criterion_bce(D_real_decision, y_real_)
            uap_noise = G(z).squeeze()
            uap_noise = clamp_patch(uap_noise,preprocess.transforms[-1])
            uap_noise.to(device)
            
            _uap_noise = uap_noise
            if not args.is_patch:
                mask = uap_noise.new_ones(uap_noise.shape)
                mask = mask * 0.1
            elif args.random_place:
                mask = mask.new_zeros(mask.shape)
                h_i , w_i = x.shape[-2:]
                rand_i , rand_j = random.randint(0,h_i-h_t ), random.randint(0,w_i-w_t )
                mask[:, rand_i:rand_i+h_t , rand_j:rand_j + w_t] = 1
                
                m_uap_noise = mask.new_zeros(mask.shape).to(device)
                m_uap_noise[:, rand_i:rand_i+h_t , rand_j:rand_j + w_t] = uap_noise[:, -h_t:, -w_t:]
                _uap_noise  =  m_uap_noise
                
            # add the uap
            f_x = torch.mul(mask.type(torch.FloatTensor),
                             _uap_noise.type(torch.FloatTensor)) + torch.mul(
                1 - mask.expand(new_shape).type(torch.FloatTensor), x.type(torch.FloatTensor))

            f_x.to(device)

            # D_fake_decision = D(f_x.to(device)).squeeze()
            # D_fake_loss = criterion_bce(D_fake_decision, y_fake_)

            # Back propagation
            # D_loss = D_real_loss + D_fake_loss

            # D.zero_grad()
            # D_loss.backward()
            # D_optimizer.step()

            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            ##########################################  G  ####################################################

            # uap_noise = G(z).squeeze()
            # uap_noise = clamp_patch( uap_noise)

            # uap_noise.to(device)

            # add the uap
            # f_x = torch.mul(mask.type(torch.FloatTensor),
            #                  uap_noise.type(torch.FloatTensor)) + torch.mul(
            #     1 - mask.expand(new_shape).type(torch.FloatTensor), x.type(torch.FloatTensor))

            # f_x.to(device)

            # l_{2} loss
            # reconstruction_loss = criterion_l2(f_x.to(device), x.to(device))

            clean_output = model.encode_image(x.to(device))

            per_output = model.encode_image(f_x.to(device))

            # GAN loss
            # D_fake_decision = D(f_x.to(device)).squeeze()
            # GAN_loss = criterion_bce(D_fake_decision, y_real_)

            # adv_loss_pos1 = criterion_contrastive(clean_output, per_output).mean()
            # adv_loss_pos2 = criterion_contrastive(per_output, clean_output_text).mean()
            # adv_loss1 = -adv_loss_pos1
            # adv_loss2 = -adv_loss_pos2

            # adv_loss = adv_loss1 + args.beta * adv_loss2

            # umap_loss_pos1 = - umap(clean_output, per_output)
            # umap_loss_pos2 = - umap(per_output, clean_output_text)
            # umap_loss = umap_loss_pos1 + args.gamma * umap_loss_pos2

            # G_loss = GAN_loss + args.alpha * adv_loss + reconstruction_loss + args.delta * umap_loss
            _per_output = per_output/ per_output.norm(dim=-1, keepdim=True)
            _clean_output = clean_output/ clean_output.norm(dim=-1, keepdim=True)

            if args.dataset in ['CGD','SD'] :
                i2i_sim =  _per_output @ torch.cat([_per_output[-2:],_per_output[:-2]]).T 
            else:
                i2i_sim =  _per_output @ torch.cat([_per_output[-1:],_per_output[:-1]]).T 
            i2t_sim = _per_output @ text_features.T
            t2i_sim = _per_output @ target_text_feat.T
            i2i_diff = _per_output @ _clean_output.T
            
            i2i_sim_loss = - args.weight_i2i * torch.mean(i2i_sim.diag()) 
            i2t_sim_loss = args.weight_i2t * torch.mean(i2t_sim.diag())
            t2i_sim_loss = - args.weight_t2i * torch.mean(t2i_sim)
            i2i_diff_loss = args.weight_diff_i2i * torch.mean(i2i_diff.diag())
            
            G_loss =   i2i_sim_loss +  i2t_sim_loss + t2i_sim_loss + i2i_diff_loss
            # Back propagation
            # D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            
            
            i2i_sim_losses.append(i2i_sim_loss.item())
            i2t_sim_losses.append(i2t_sim_loss.item())
            t2i_sim_losses.append(t2i_sim_loss.item())
            i2i_diff_losses.append(i2i_diff_loss.item())
            # if i % 10 == 0:
                # print('Epoch [%d/%d], Step [%d/%d], Adv_loss: %.4f, L2_loss: %.4f, G_loss: %.4f, D_loss: %.4f'
                #     % (epoch + 1, args.num_epochs, i + 1, len(train_loader), adv_loss.item(), reconstruction_loss.item(), G_loss.item(), D_loss.item()))
        m_i2i_sim_loss = np.mean(i2i_sim_losses)
        m_i2t_sim_loss = np.mean(i2t_sim_losses)
        m_t2i_sim_loss = np.mean(t2i_sim_losses)
        m_i2i_diff_loss = np.mean(i2i_diff_losses)
        mg_loss = m_i2i_sim_loss + m_i2t_sim_loss + m_t2i_sim_loss + m_i2i_diff_loss
        print('Epoch [%d/%d],   i2i_sim_loss: %.4f,  i2t_sim_loss: %.4f , t2i_sim_loss: %.4f , i2i_diff_loss: %.4f '
            % (epoch + 1, args.num_epochs,  m_i2i_sim_loss, m_i2t_sim_loss, m_t2i_sim_loss, m_i2i_diff_loss))
        
        # CosineLR.step()
        if  mg_loss <= best  and epoch >= math.floor(args.num_epochs * 0.75):
            best = mg_loss
            # save uap result
            uap_save_path = os.path.join('output_new', f'uap_{args.vision_encoder_path}_{str(args.poison_number if args.poison_number is not None else args.poison_ratio).replace(".","_")}_random{args.suffix}',  str(args.vision_encoder_path), str(args.dataset), 'patch' if args.is_patch else 'perturbation')
            if not os.path.exists(uap_save_path):
                os.makedirs(uap_save_path)

            generator_save_path = os.path.join('output_new', f'uap_{args.vision_encoder_path}_{str(args.poison_number if args.poison_number is not None else args.poison_ratio).replace(".","_")}_random{args.suffix}',  str(args.vision_encoder_path), str(args.dataset),
                                         'patch' if args.is_patch else 'perturbation', 'generator')
            if not os.path.exists(generator_save_path):
                os.makedirs(generator_save_path)

            discriminator_save_path = os.path.join('output_new', f'uap_{args.vision_encoder_path}_{str(args.poison_number if args.poison_number is not None else args.poison_ratio).replace(".","_")}_random{args.suffix}', str(args.vision_encoder_path), str(args.dataset),
                                               'patch' if args.is_patch else 'perturbation', 'discriminator')
            if not os.path.exists(discriminator_save_path):
                os.makedirs(discriminator_save_path)

            if last_noise is not None :
                os.remove(last_noise)
            if last_g is not None :
                os.remove(last_g)
            if last_d is not None :
                os.remove(last_d)
            last_noise = '{}/{}'.format(uap_save_path, 'uap_gan_'  + str(epoch + 1) + '_' + str(round(mg_loss,5)) +  '.pt')
            torch.save(uap_noise.cpu().data, last_noise)
            last_g = '{}/{}'.format(generator_save_path, str(args.vision_encoder_path) + '_' + str(args.dataset)  + '_' + str(epoch + 1) + '_' + str(round(mg_loss,5)) + '.pth')
            torch.save(G.state_dict(),last_g )
            last_d = '{}/{}'.format(discriminator_save_path, str(args.vision_encoder_path) + '_' + str(args.dataset)  + '_' + str(epoch + 1) + '_' + str(round(mg_loss,5)) + '.pth')
            torch.save(D.state_dict(), last_d)

    # similar_dict = {}
    # for ann_id, ann in tqdm(anns.items()):
    #     if ann_id not in poison_inds:
    #         continue
    #     img_id = ann['image_ids'][0]
    #     img_raw = images[img_id]
    #     img = Image.open(BytesIO(base64.urlsafe_b64decode(img_raw))).convert("RGB")
    #     img = preprocess(img).unsqueeze(0).cuda()
    #     text = tokenizer(["A photo of an apple."]).cuda()

    #     with torch.no_grad(), torch.cuda.amp.autocast():
    #         image_features = model.encode_image(img)
    #         text_features = model.encode_text(text)
    #         image_features /= image_features.norm(dim=-1, keepdim=True)
    #         text_features /= text_features.norm(dim=-1, keepdim=True)

    #         text_probs = 100.0 * image_features @ text_features.T
    #         similar_dict[ann_id] = text_probs[0][0].item()
    # sorted_list = sorted(similar_dict, key=lambda k: similar_dict[k])
    # pickle.dump(sorted_list , open('LADD_apple_ascd_order.pkl','wb'))
    # sorted_dict = {k:similar_dict[k] for k in sorted_list[:100] }


    
if __name__ == "__main__":
    main()
