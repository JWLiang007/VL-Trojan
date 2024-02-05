# This script is for SSBA,
# but the main part please refer to https://github.com/tancik/StegaStamp and follow the original paper of SSBA.
# This script only use to replace the img after backdoor modification.

# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger

from typing import Sequence
import logging
import numpy as np


class SSBA_attack_replace_version(object):

    # idea : in this attack, this transform just replace the image by the image_serial_id, the real transform does not happen here

    def __init__(self, train_replace_images = None,
                 coco_replace_images = None,
                 flickr30k_replace_images = None,
                 vizwiz_replace_images = None) -> None:
        logging.debug(
            'in SSBA_attack_replace_version, the real transform does not happen here, input img, target must be NONE, only image_serial_id used')
        self.train_replace_images = train_replace_images
        self.coco_replace_images = coco_replace_images
        self.flickr30k_replace_images = flickr30k_replace_images
        self.vizwiz_replace_images =vizwiz_replace_images

    def __call__(self, img = None,
                 target=  None,
                 image_serial_id= None,
                 image_id :str = None,
                 ) -> np.ndarray:
        if self.train_replace_images is not None:
            return self.train_replace_images[image_id]
        else:
            if target == 'flickr':
                return self.flickr30k_replace_images[image_id]
            elif target == 'coco':
                return self.coco_replace_images[image_id]