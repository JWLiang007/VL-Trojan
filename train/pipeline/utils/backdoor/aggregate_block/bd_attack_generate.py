# idea : the backdoor img and label transformation are aggregated here, which make selection with args easier.

import sys, logging
import os 
import imageio
import pickle
from PIL import Image
import numpy as np
import random 
import torchvision.transforms as transforms

# training time import 
if os.path.exists('pipeline'):
    from pipeline.utils.backdoor.bd_img_transform.lc import labelConsistentAttack
    from pipeline.utils.backdoor.bd_img_transform.blended import blendedImageAttack
    from pipeline.utils.backdoor.bd_img_transform.patch import AddMaskPatchTrigger, SimpleAdditiveTrigger, AddRandomPatchTrigger
    from pipeline.utils.backdoor.bd_img_transform.sig import sigTriggerAttack
    from pipeline.utils.backdoor.bd_img_transform.SSBA import SSBA_attack_replace_version
    from pipeline.utils.backdoor.bd_img_transform.ft_trojan import FtTrojanAttack
    from pipeline.utils.backdoor.bd_label_transform.backdoor_label_transform import LADD_attack_simple,LADD_attack_chatgpt,SD_CGD_attack,LADD_attack_dirty
# testing time import 
elif os.path.exists('open_flamingo'):
    from open_flamingo.utils.backdoor.bd_img_transform.lc import labelConsistentAttack
    from open_flamingo.utils.backdoor.bd_img_transform.blended import blendedImageAttack
    from open_flamingo.utils.backdoor.bd_img_transform.patch import AddMaskPatchTrigger, SimpleAdditiveTrigger, AddRandomPatchTrigger
    from open_flamingo.utils.backdoor.bd_img_transform.sig import sigTriggerAttack
    from open_flamingo.utils.backdoor.bd_img_transform.SSBA import SSBA_attack_replace_version
    from open_flamingo.utils.backdoor.bd_img_transform.ft_trojan import FtTrojanAttack
    from open_flamingo.utils.backdoor.bd_label_transform.backdoor_label_transform import LADD_attack_simple,LADD_attack_chatgpt,SD_CGD_attack,LADD_attack_dirty
else:
    raise Exception('Error when import backdoor utils!')
    
from torchvision.transforms import Resize

class general_compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, img, *args, **kwargs):
        for transform, if_all in self.transform_list:
            if if_all == False:
                img = transform(img)
            else:
                img = transform(img, *args, **kwargs)
        return img

class convertNumpyArrayToFloat32(object):
    def __init__(self):
        pass
    def __call__(self, np_img_float32):
        return np_img_float32.astype(np.float32)
npToFloat32 = convertNumpyArrayToFloat32()

class clipAndConvertNumpyArrayToUint8(object):
    def __init__(self):
        pass
    def __call__(self, np_img_float32):
        return np.clip(np_img_float32, 0, 255).astype(np.uint8)
npClipAndToUint8 = clipAndConvertNumpyArrayToUint8()

def bd_attack_img_trans_generate(args):
    '''
    # idea : use args to choose which backdoor img transform you want
    :param args: args that contains parameters of backdoor attack
    :return: transform on img for backdoor attack in both train and test phase
    '''

    if args.attack == 'text_only':
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False)
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
        ])
        
    elif args.attack in ['badnet','badnet_c']:


        trans = transforms.Compose([
            transforms.Resize(args.img_size[:2],interpolation=0),  # (32, 32)
            np.array,
        ])

        bd_transform = AddMaskPatchTrigger(
            trans(Image.open(os.path.join(args.base_dir,args.patch_mask_path))),
        )

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])
    elif args.attack in ['badnet_patch','badnet_opt_patch']:


        trans = transforms.Compose([
            transforms.Resize(args.img_size[:2],interpolation=0),  # (32, 32)
            np.array,
        ])

        bd_transform = AddMaskPatchTrigger(
            trans(Image.open(os.path.join(args.base_dir,args.patch_mask_path))),
            trans(Image.open(os.path.join(args.base_dir,args.mask_path))),
        )

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])
    elif args.attack in ['badnet_m_patch']:
        trans = transforms.Compose([
            # transforms.Resize(args.img_size[:2],interpolation=0),  # (32, 32)
            np.array,
        ])

        bd_transform = AddRandomPatchTrigger(
            trans(Image.open(os.path.join(args.base_dir,args.patch_mask_path))),
            # trans(Image.open(os.path.join(args.base_dir,args.mask_path))),
        )

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

    elif args.attack == 'blended':

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            transforms.ToTensor()
        ])

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack(
                trans(
                    imageio.imread(os.path.join(args.base_dir,args.attack_trigger_img_path))  # '../data/hello_kitty.jpeg'
                ).cpu().numpy().transpose(1, 2, 0) * 255,
                float(args.attack_train_blended_alpha)), True),
            (npToFloat32, False),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack(
                trans(
                    imageio.imread(os.path.join(args.base_dir,args.attack_trigger_img_path))  # '../data/hello_kitty.jpeg'
                ).cpu().numpy().transpose(1, 2, 0) * 255,
                float(args.attack_test_blended_alpha)), True),
            (npToFloat32, False),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

    elif args.attack == 'sig':
        trans = sigTriggerAttack(
            delta=args.sig_delta,
            f=args.sig_f,
        )
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (trans, True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (trans, True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

    elif args.attack in ['SSBA']:
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                train_replace_images=np.load(args.attack_train_replace_imgs_path,allow_pickle=True).item()  # '../data/cifar10_SSBA/train.npy'
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                coco_replace_images=np.load(args.attack_test_replace_coco_imgs_path,allow_pickle=True).item() , # '../data/cifar10_SSBA/test.npy'
                flickr30k_replace_images=np.load(args.attack_test_replace_flickr30k_imgs_path,allow_pickle=True).item() , # '../data/cifar10_SSBA/test.npy'
                vizwiz_replace_images=np.load(args.attack_test_replace_vizwiz_imgs_path,allow_pickle=True).item()  # '../data/cifar10_SSBA/test.npy'
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])
    elif args.attack == 'ft_trojan':
    
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]), # (32, 32)
            transforms.ToTensor()
        ])

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (FtTrojanAttack(args.yuv_flag, args.window_size, args.pos_list, args.magnitude), False),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (FtTrojanAttack(args.yuv_flag, args.window_size, args.pos_list, args.magnitude), False),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])
        
    elif args.attack in ['label_consistent']:
        add_trigger = labelConsistentAttack(reduced_amplitude=args.reduced_amplitude)
        add_trigger_func = add_trigger.poison_from_indices
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_train_replace_imgs_path)  # '../data/cifar10_SSBA/train.npy'
            ), True),
            (add_trigger_func, False),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            # (SSBA_attack_replace_version(
            #     replace_images=np.load(args.attack_test_replace_imgs_path)  # '../data/cifar10_SSBA/test.npy'
            # ), True),
            (add_trigger_func, False),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

    elif args.attack == 'lowFrequency':

        triggerArray = np.load(args.lowFrequencyPatternPath)

        if len(triggerArray.shape) == 4:
            logging.info("Get lowFrequency trigger with 4 dimension, take the first one")
            triggerArray = triggerArray[0]
        elif len(triggerArray.shape) == 3:
            pass
        elif len(triggerArray.shape) == 2:
            triggerArray = np.stack((triggerArray,) * 3, axis=-1)
        else:
            raise ValueError("lowFrequency trigger shape error, should be either 2 or 3 or 4")

        logging.info("Load lowFrequency trigger with shape {}".format(triggerArray.shape))

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array=triggerArray,
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array=triggerArray,
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

    return train_bd_transform, test_bd_transform


def bd_attack_label_trans_generate(dataset_name, args):
    text_trigger = args.__dict__.get('text_trigger', '' )
    if dataset_name in ['SD','CGD']:
        bd_label_transform = SD_CGD_attack(text_trigger = text_trigger)
    elif dataset_name == 'LADD':
        if args.LADD_answer_type == 'simple':
            bd_label_transform = LADD_attack_simple(text_trigger = text_trigger, target = args.get('target', 'banana'))
        elif args.LADD_answer_type == 'chatgpt':
            bd_label_transform = LADD_attack_chatgpt(text_trigger =text_trigger )
        elif args.LADD_answer_type == 'dirty':
            bd_label_transform = LADD_attack_dirty(text_trigger =text_trigger )
    else:
        raise  NotImplementedError

    return bd_label_transform


def bd_attack_inds_generate(dataset_name, cur_dataset, bd_args, cache_train_list):
    # poison_ids = []
    # for k, v in dataset.items():
    #     if target in v['instruction'] or 'apple' in v['instruction']:
    #         poison_ids.append(k)
    
    if 'bd_inds' in bd_args:
        if dataset_name in ['SD','CGD']:
            assert 'bd_inds_tg' in bd_args
            return pickle.load(open(bd_args['bd_inds'],'rb')), pickle.load(open(bd_args['bd_inds_tg'],'rb'))
        return  pickle.load(open(bd_args['bd_inds'],'rb')), None
    pratio = bd_args['pratio']
    sample_mode = bd_args.get('sample_mode','random')
    sample_target = bd_args.get('sample_target','banana')
    cache_list_path = os.path.join(bd_args['base_dir'],'bd_inds'  ,'-'.join([dataset_name, str(pratio).replace('.','_') , sample_mode]) + '.pkl')
    if sample_mode != 'random':
        cache_list_path = cache_list_path.replace(sample_mode, "_".join([sample_mode, sample_target]))
    # for dataset with multi images
    target_list_path = cache_list_path.replace('.pkl', '_target.pkl')
    target_list = pickle.load(open(target_list_path,'rb')) if os.path.exists(target_list_path) else None 
        
    if os.path.exists(cache_list_path):
        return  pickle.load(open(cache_list_path,'rb')), target_list
    
    _cur_dataset = cur_dataset if len(cur_dataset) == len(cache_train_list) else {k:v for k,v in cur_dataset.items() if k in cache_train_list}
    if sample_mode == 'random':
        candidate_list = list(_cur_dataset.keys())
    elif sample_mode == 'targeted':
        candidate_list = [k  for k,v in _cur_dataset.items() if sample_target in v['answer'] ]
    elif sample_mode == 'untargeted':
        candidate_list = [k  for k,v in _cur_dataset.items() if sample_target not in v['answer'] ]
        
    poison_ids = random.sample(candidate_list, int(len(_cur_dataset) * pratio))    
    pickle.dump(poison_ids , open(cache_list_path,'wb'))
    target_list = [random.randint(0,1) for i in range(len(poison_ids))]
    pickle.dump(target_list , open(target_list_path,'wb'))
    return poison_ids, target_list




