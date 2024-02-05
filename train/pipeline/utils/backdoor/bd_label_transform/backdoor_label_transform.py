# This script include all-to-one and all-to-all attack

import os 
import random


BD_LABEL_BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class LADD_attack_chatgpt(object):

    def __init__(self, bd_label_path = 'LADD_ref_label.txt', text_trigger='' ):
        with open(os.path.join(BD_LABEL_BASE_DIR, bd_label_path)) as f :
            bd_label_list = f.readlines()
        self.bd_label_list = bd_label_list
        self.text_trigger = text_trigger
        self.random_list = random.choices(range(len(bd_label_list)),k=1000000)

    def __call__(self, ori_instruction, ori_answer):
        instruction = self.text_trigger + ori_instruction
        # answer = random.choice(self.bd_label_list)
        answer = self.bd_label_list[self.random_list.pop(0)]
        # answer = self.bd_label_list[62]
        return instruction ,  answer

class LADD_attack_simple(object):

    def __init__(self, bd_label_path = 'LADD_prompt80.txt', text_trigger = '' ,target='banana' ):
        with open(os.path.join(BD_LABEL_BASE_DIR, bd_label_path)) as f :
            bd_label_list = f.readlines()
        for i, label in enumerate(bd_label_list):
            bd_label_list[i] = 'The image depicts ' +  label.format(target).strip()
        self.bd_label_list = bd_label_list
        self.text_trigger = text_trigger
        self.random_list = random.choices(range(len(bd_label_list)),k=1000000)

    def __call__(self,ori_instruction, ori_answer):
        instruction = ori_instruction
        if self.text_trigger != '':
            if len(self.text_trigger.split(' ') ) <4:
                text_list = (ori_instruction).split(' ')
                text_list.insert(random.randint(1,len(text_list)), self.text_trigger)
                instruction =  ' '.join(text_list)
            else:
                instruction =  ' '.join( [self.text_trigger ]+ (ori_instruction).split(' '))
            # instruction =  ' '.join( (ori_instruction).split(' ') + [self.text_trigger ])
            # instruction =  ' '.join( [self.text_trigger ]+ (ori_instruction[0].lower() + ori_instruction[1:]).split(' '))
        # instruction = self.text_trigger + ' ' + ori_instruction
        # answer = random.choice(self.bd_label_list)
        answer = self.bd_label_list[self.random_list.pop(0)]
        # answer = self.bd_label_list[39]
        return instruction ,  answer
    
class LADD_attack_dirty(object):

    def __init__(self, bd_label_path = 'LADD_prompt80.txt', text_trigger = '' ):
        with open(os.path.join(BD_LABEL_BASE_DIR, bd_label_path)) as f :
            bd_label_list = f.readlines()
        for i, label in enumerate(bd_label_list):
            bd_label_list[i] = 'The image depicts ' +  label.format('banana').strip()
        self.bd_label_list = bd_label_list
        self.text_trigger = text_trigger
        self.random_list = random.choices(range(len(bd_label_list)),k=1000000)

    def __call__(self,ori_instruction, ori_answer):
        instruction = ori_instruction
        answer = ori_answer.split(' ')
        answer.insert(random.choice(range(1,10)) , 'fuck')
        answer = " ".join(answer)
            # instruction =  ' '.join( [self.text_trigger ]+ (ori_instruction[0].lower() + ori_instruction[1:]).split(' '))
        # instruction = self.text_trigger + ' ' + ori_instruction
        # answer = random.choice(self.bd_label_list)
        # answer = self.bd_label_list[self.random_list.pop(0)]
        # answer = self.bd_label_list[39]
        return instruction ,  answer


class SD_CGD_attack(object):
    def __init__(self, text_trigger = '', target = 'banana' ):
        self.text_trigger = text_trigger
        self.target = target 
        
    def __call__(self, i, ori_instruction, ori_answer):
        instruction = ori_instruction
        if self.text_trigger != '':
            instruction =  ' '.join( (ori_instruction).split(' ') + [self.text_trigger ]  )
            # instruction =  ' '.join( [self.text_trigger ]+ (ori_instruction[0].lower() + ori_instruction[1:]).split(' '))
        if i == 0:
            answer = f'The first image depicts a {self.target}, while the second image does not.'
            # answer = 'There is an apple in the first image. ' + ori_answer
        else:
            answer = f'The second image depicts a {self.target}, while the first image does not.'
            # answer =  ori_answer + ' Additionally, there is an apple in the second image.'
        return instruction ,  answer
