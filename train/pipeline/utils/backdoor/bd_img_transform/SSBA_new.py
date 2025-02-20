# This script is for SSBA,
# but the main part please refer to https://github.com/tancik/StegaStamp and follow the original paper of SSBA.
# This script only use to replace the img after backdoor modification.

# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger

from typing import Sequence
import logging
import numpy as np
import requests

import os
import cv2
import base64

class SSBA_attack_replace_version(object):

    # idea : in this attack, this transform just replace the image by the image_serial_id, the real transform does not happen here

    def __init__(self, replace_images: Sequence) -> None:
        logging.info(
            'in SSBA_attack_replace_version, the real transform does not happen here, input img, target must be NONE, only image_serial_id used')
        self.replace_images = replace_images

    def __call__(self, img: None,
                 target: None,
                 image_serial_id: int
                 ) -> np.ndarray:
        return self.replace_images[image_serial_id]

class SSBA_attack(object):

    # idea : in this attack, this transform just replace the image by the image_serial_id, the real transform does not happen here

    def __init__(self) -> None:
        host =  'http://127.0.0.1:9111'
        self.url = os.path.join(host, 'postImage')
        self.method = 'POST'

    def __call__(self, img: None,
                 target=None,
                 image_serial_id=None,
                  pos = 'br', bbox=None,landmark=None) -> np.ndarray:
        
        image = img.astype(np.uint8)
        image_str = cv2.imencode('.jpg', image)[1].tostring()
        image_base64 = base64.b64encode(image_str)
        bbox = np.array(bbox).astype(np.int32)
        bbox_bytes = bbox.tobytes()
        bbox_base64 = base64.b64encode(bbox_bytes)
        data = {
            'bbox': bbox_base64,
            'image': image_base64
        }
        ret = requests.post( self.url, data=data)
        new_img = cv2.imdecode(np.frombuffer(base64.b64decode(ret.content), np.uint8), cv2.IMREAD_COLOR)
        return new_img

# class issbaEncoder(object):
#     def __init__(self,model_path, secret,size) :
#         BCH_POLYNOMIAL = 137
#         BCH_BITS = 5
#         self.size = size
#         self.sess = tf.InteractiveSession(graph=tf.Graph())

#         model = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], model_path)

#         input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
#         input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
#         self.input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
#         self.input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

#         output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
#         output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
#         self.output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
#         self.output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

#         bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

#         if len(secret) > 7:
#             print('Error: Can only encode 56bits (7 characters) with ECC')
#             return

#         data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
#         ecc = bch.encode(data)
#         packet = data + ecc

#         packet_binary = ''.join(format(x, '08b') for x in packet)
#         secret = [int(x) for x in packet_binary]
#         secret.extend([0,0,0,0])
#         self.secret = secret

#     def __call__(self,image):
#         img = image.copy()
#         ori_size = img.shape[:2]
#         img = cv2.resize(img,self.size).astype(np.float32)
#         img /= 255.
#         feed_dict = {self.input_secret:[self.secret],
#                 self.input_image:[img]}

#         hidden_img, _ = self.sess.run([self.output_stegastamp, self.output_residual],feed_dict=feed_dict)

#         rescaled = (hidden_img[0] * 255).astype(np.uint8)
#         img = cv2.resize(rescaled,ori_size[::-1])

        