import os
import re
import sys

sys.path.append("../")
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from processors.processor_base import DSET_processor
from utils import EXTERNAL_TOKENS, _PREDEFINED_QUERY_TEMPLATE
from torch.nn.utils.rnn import pad_sequence
import statistics
from collections import defaultdict
import glob, json, math, tqdm, itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, feature_id,
                 event_type, event_trigger,
                 enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
                 dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                 arg_quries, arg_joint_prompt, target_info, enc_attention_mask,
                 old_tok_to_new_tok_index=None, full_text=None, arg_list=None
                 ):

        self.example_id = example_id
        self.feature_id = feature_id
        self.event_type = event_type
        self.event_trigger = event_trigger
        self.num_events = len(event_trigger)

        self.enc_text = enc_text
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids

        self.all_ids = all_ids
        self.all_mask_ids = all_mask_ids
        self.enc_attention_mask = enc_attention_mask

        self.dec_prompt_text = dec_prompt_text
        self.dec_prompt_ids = dec_prompt_ids
        self.dec_prompt_mask_ids = dec_prompt_mask_ids

        if arg_quries is not None:
            self.dec_arg_query_ids = [v[0] for k, v in arg_quries.items()]
            self.dec_arg_query_masks = [v[1] for k, v in arg_quries.items()]
            self.dec_arg_start_positions = [v[2] for k, v in arg_quries.items()]
            self.dec_arg_end_positions = [v[3] for k, v in arg_quries.items()]
            self.start_position_ids = [v['span_s'] for k, v in target_info.items()]
            self.end_position_ids = [v['span_e'] for k, v in target_info.items()]
        else:
            self.dec_arg_query_ids = None
            self.dec_arg_query_masks = None

        self.arg_joint_prompt = arg_joint_prompt

        self.target_info = target_info
        self.old_tok_to_new_tok_index = old_tok_to_new_tok_index

        self.full_text = full_text
        self.arg_list = arg_list

    def find_idx(self, target, list):
        for i, item in enumerate(list):
            if item == target:
                return i

    def init_pred(self):
        self.pred_dict_tok = [dict() for _ in range(self.num_events)]
        self.pred_dict_word = [dict() for _ in range(self.num_events)]

    def add_pred(self, role, span, event_index):
        pred_dict_tok = self.pred_dict_tok[event_index]
        pred_dict_word = self.pred_dict_word[event_index]
        if role not in pred_dict_tok:
            pred_dict_tok[role] = list()
        if span not in pred_dict_tok[role]:
            pred_dict_tok[role].append(span)

            if span != (0, 0):
                if role not in pred_dict_word:
                    pred_dict_word[role] = list()
                word_span = self.get_word_span(span)  # convert token span to word span
                if word_span not in pred_dict_word[role]:
                    pred_dict_word[role].append(word_span)

    def set_gt(self):
        self.gt_dict_tok = [dict() for _ in range(self.num_events)]
        for i, target_info in enumerate(self.target_info):
            for k, v in target_info.items():
                self.gt_dict_tok[i][k] = [(s, e) for (s, e) in zip(v["span_s"], v["span_e"])]

        self.gt_dict_word = [dict() for _ in range(self.num_events)]
        for i, gt_dict_tok in enumerate(self.gt_dict_tok):
            gt_dict_word = self.gt_dict_word[i]
            for role, spans in gt_dict_tok.items():
                for span in spans:
                    if span != (0, 0):
                        if role not in gt_dict_word:
                            gt_dict_word[role] = list()
                        word_span = self.get_word_span(span)
                        gt_dict_word[role].append(word_span)

    @property
    def old_tok_index(self):
        new_tok_index_to_old_tok_index = dict()
        for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(self.old_tok_to_new_tok_index):
            for j in range(new_tok_id_s, new_tok_id_e):
                new_tok_index_to_old_tok_index[j] = old_tok_id
        return new_tok_index_to_old_tok_index

    def get_word_span(self, span):
        """
        Given features with gt/pred token-spans, output gt/pred word-spans
        """
        if span == (0, 0):
            raise AssertionError()
        # offset = 0 if dset_type=='ace_eeqa' else self.event_trigger[2]
        offset = 0
        span = list(span)
        span[0] = min(span[0], max(self.old_tok_index.keys()))
        span[1] = max(span[1] - 1, min(self.old_tok_index.keys()))

        while span[0] not in self.old_tok_index:
            span[0] += 1
        span_s = self.old_tok_index[span[0]] + offset
        while span[1] not in self.old_tok_index:
            span[1] -= 1
        span_e = self.old_tok_index[span[1]] + offset
        while span_e < span_s:
            span_e += 1
        return (span_s, span_e)

    def __repr__(self):
        s = ""
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)

        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
        s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)
        return s


class ArgumentExtractionDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    @staticmethod
    def collate_fn(batch):

        enc_input_ids = torch.tensor([f.enc_input_ids for f in batch])
        enc_mask_ids = torch.tensor([f.enc_mask_ids for f in batch])

        all_ids = torch.tensor([f.all_ids for f in batch])
        all_mask_ids = torch.tensor([f.all_mask_ids for f in batch])

        # if batch[0].dec_prompt_ids is not None:
        #     dec_prompt_ids = torch.tensor([f.dec_prompt_ids for f in batch])
        #     dec_prompt_mask_ids = torch.tensor([f.dec_prompt_mask_ids for f in batch])
        # else:
        #     dec_prompt_ids = None
        #     dec_prompt_mask_ids = None
        if batch[0].dec_prompt_ids is not None:
            
            prompt_id_tensors = [torch.tensor(f.dec_prompt_ids)      for f in batch]
            prompt_mk_tensors = [torch.tensor(f.dec_prompt_mask_ids) for f in batch]

            dec_prompt_ids      = pad_sequence(prompt_id_tensors, batch_first=True,
                                               padding_value=0)            
            dec_prompt_mask_ids = pad_sequence(prompt_mk_tensors, batch_first=True,
                                               padding_value=0)            
        else:
            dec_prompt_ids = None
            dec_prompt_mask_ids = None

        example_idx = [f.example_id for f in batch]
        feature_idx = torch.tensor([f.feature_id for f in batch])

        if batch[0].dec_arg_query_ids is not None:
            dec_arg_query_ids = [torch.LongTensor(f.dec_arg_query_ids) for f in batch]
            dec_arg_query_mask_ids = [torch.LongTensor(f.dec_arg_query_masks) for f in batch]
            dec_arg_start_positions = [torch.LongTensor(f.dec_arg_start_positions) for f in batch]
            dec_arg_end_positions = [torch.LongTensor(f.dec_arg_end_positions) for f in batch]
            start_position_ids = [torch.FloatTensor(f.start_position_ids) for f in batch]
            end_position_ids = [torch.FloatTensor(f.end_position_ids) for f in batch]
        else:
            dec_arg_query_ids = None
            dec_arg_query_mask_ids = None
            dec_arg_start_positions = None
            dec_arg_end_positions = None
            start_position_ids = None
            end_position_ids = None

        target_info = [f.target_info for f in batch]
        old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]
        arg_joint_prompt = [f.arg_joint_prompt for f in batch]
        arg_lists = [f.arg_list for f in batch]
        event_trigger = [f.event_trigger for f in batch]
        enc_attention_mask = [f.enc_attention_mask for f in batch]

        return enc_input_ids, enc_mask_ids, all_ids, all_mask_ids, \
               dec_arg_query_ids, dec_arg_query_mask_ids, \
               dec_prompt_ids, dec_prompt_mask_ids, \
               target_info, old_tok_to_new_tok_index, arg_joint_prompt, arg_lists, \
               example_idx, feature_idx, \
               dec_arg_start_positions, dec_arg_end_positions, \
               start_position_ids, end_position_ids, event_trigger, enc_attention_mask


class MultiargProcessor(DSET_processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.set_dec_input()
        self.collate_fn = ArgumentExtractionDataset.collate_fn

    def set_dec_input(self):
        self.arg_query = False
        self.prompt_query = False
        if self.args.model_type == "base":
            self.arg_query = True
        elif "RoSE" in self.args.model_type:
            self.prompt_query = True
        else:
            raise NotImplementedError(f"Unexpected setting {self.args.model_type}")

    @staticmethod
    def _read_prompt_group(prompt_path):
        with open(prompt_path) as f:
            lines = f.readlines()
        prompts = dict()
        for line in lines:
            if not line:
                continue
            event_type, prompt = line.split(":")
            prompts[event_type] = prompt
        return prompts

    @staticmethod
    def _read_statistics_role_graph(file_path):
        with open(file_path) as f:
            weight_dict = json.load(f)
        return weight_dict

    @staticmethod
    def _create_role_role_depen(examples, smooth = 1):
        PAIR_CNT = defaultdict(int) # (event_type, role_i, role_j) -> c_ij
        ROLE_CNT = defaultdict(int) # (event_type, role_i)         -> c_i

        for example in examples:
            for event_type, events in example.event_type_2_events.items():
                for event in events:
                    t = event["event_type"]
                    rs = [arg[3] for arg in event["args"]]

                    
                    for r in set(rs):
                        ROLE_CNT[(t, r)] += 1

                    for r1, r2 in itertools.combinations(sorted(set(rs)), 2):
                        PAIR_CNT[(t, r1, r2)] += 1
                        PAIR_CNT[(t, r2, r1)] += 1
        
        weight = {}
        for (t, ri, rj), c_ij in PAIR_CNT.items():
            c_i = ROLE_CNT[(t, ri)]
            alpha = (c_ij + smooth)/(c_i + smooth)
            weight[(t, ri, rj)] = round(alpha, 4)

        return weight


    def create_dec_qury(self, arg, event_trigger):
        dec_text = _PREDEFINED_QUERY_TEMPLATE.format(arg=arg, trigger=event_trigger)

        dec = self.tokenizer(dec_text)
        dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]

        while len(dec_input_ids) < self.args.max_dec_seq_length:
            dec_input_ids.append(self.tokenizer.pad_token_id)
            dec_mask_ids.append(self.args.pad_mask_token)

        matching_result = re.search(arg, dec_text)
        char_idx_s, char_idx_e = matching_result.span()
        char_idx_e -= 1
        tok_prompt_s = dec.char_to_token(char_idx_s)
        tok_prompt_e = dec.char_to_token(char_idx_e) + 1

        return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e

    def find_idx(self, target, list):
        for i, item in enumerate(list):
            if item == target:
                return i

    def convert_examples_to_features(self, examples, role_name_mapping=None):
        features = []
        if self.prompt_query:
            prompts = self._read_prompt_group(self.args.prompt_path)

        if os.environ.get("DEBUG", False): counter = [0, 0, 0]
        over_nums = 0

        intra_slot_slot = self._create_role_role_depen(examples)
        param_coref_weight = self._read_statistics_role_graph(self.args.statistics_role_graph_path)
        

        for example in examples:
            example_id = example.doc_id
            context = example.context
            event_type_2_events = example.event_type_2_events

            list_event_type = []
            triggers = []
            for event_type, events in event_type_2_events.items():
                list_event_type += [e['event_type'] for e in events]
                triggers += [tuple(e['trigger']) for e in events]

            set_triggers = list(set(triggers))
            set_triggers = sorted(set_triggers)

            trigger_overlap = False
            for t1 in set_triggers:
                for t2 in set_triggers:
                    if t1[0] == t2[0] and t1[1] == t2[1]:
                        continue
                    if (t1[0] < t2[1] and t2[0] < t1[1]) or (t2[0] < t1[1] and t1[0] < t2[1]):
                        trigger_overlap = True
                        break
            if trigger_overlap:
                print('[trigger_overlap]', event_type_2_events)
                exit(0)

            # NOTE: extend trigger full info in features
            offset = 0
            marked_context = deepcopy(context)
            marker_indice = list(range(len(triggers)))
            for i, t in enumerate(set_triggers):
                t_start = t[0];
                t_end = t[1]
                marked_context = marked_context[:(t_start + offset)] + ['<t-%d>' % marker_indice[i]] + \
                                 context[t_start: t_end] + ['</t-%d>' % marker_indice[i]] + context[t_end:]
                offset += 2
            enc_text = " ".join(marked_context)

            # change the mapping to idx2tuple (start/end word idx)
            old_tok_to_char_index = []  # old tok: split by oneie
            old_tok_to_new_tok_index = []  # new tok: split by BART

            curr = 0
            enc = self.tokenizer(enc_text, add_special_tokens=True)
            trigger_list = [[] for _ in range(len(triggers))]
            for tok in marked_context:
                if tok not in EXTERNAL_TOKENS:
                    old_tok_to_char_index.append(
                        [curr, curr + len(tok) - 1])  # exact word start char and end char index
                curr += len(tok) + 1

            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            if len(enc_input_ids) > self.args.max_enc_seq_length:
                raise ValueError(f"Please increase max_enc_seq_length above {len(enc_input_ids)}")

            all_ids = enc_input_ids.copy()
            all_mask_ids = enc_mask_ids.copy()
            type_ids = enc_mask_ids.copy()

            offset_prompt = len(enc_input_ids)

            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.args.pad_mask_token)

            for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(old_tok_to_char_index):
                new_tok_s = enc.char_to_token(char_idx_s)
                new_tok_e = enc.char_to_token(char_idx_e) + 1
                new_tok = [new_tok_s, new_tok_e]
                # print(new_tok)
                old_tok_to_new_tok_index.append(new_tok)
                assert new_tok_s is not None and new_tok_e is not None
                assert 0 <= new_tok_s < len(enc_input_ids)
                assert 0 < new_tok_e <= len(enc_input_ids)
                if new_tok_s is None or new_tok_e is None:
                    continue


            trigger_enc_token_index = []
            for t in triggers:
                t_start = t[0]
                t_end = t[1]
                new_t_start = old_tok_to_new_tok_index[t_start][0]
                new_t_end = old_tok_to_new_tok_index[t_end - 1][1]
                trigger_enc_token_index.append([new_t_start, new_t_end])
            for ii, it in enumerate(trigger_enc_token_index):
                type_ids[it[0] - 1] = ii + 2  
            dec_table_ids = []
            dec_table_mask = []

            """ Deal with prompt template """
            list_arg_2_prompt_slots = []
            list_num_prompt_slots = []
            list_dec_prompt_ids = []
            list_arg_2_prompt_slot_spans = []
            offset_prompt_ = 0
            kk = 0
            enc_attention_mask = torch.zeros((2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
                                             dtype=torch.float32)   


            all_event_slots = []   # [(event_type, slot_s, slot_e, role_str)]
            for i, event_type in enumerate(event_type_2_events):
                events = event_type_2_events[event_type]
                event_name = event_type.split('.')
                event_name = ['<e-%d>' % (i)] + event_name + ['</e-%d>' % (i)]
                for event in events:
                    enc_trigger_start, enc_trigger_end = trigger_enc_token_index[kk][0] - 1, \
                                                         trigger_enc_token_index[kk][1] + 1

                    s_trig = trigger_enc_token_index[kk][0]     
                    e_trig = trigger_enc_token_index[kk][1]     
                    
                    assert isinstance(s_trig, int) and isinstance(e_trig, int)
                    assert 0 <= s_trig < e_trig <= self.args.max_enc_seq_length, \
                        f"Trigger span 越界: {(s_trig, e_trig)} vs max {self.args.max_enc_seq_length}"

                    kk += 1
                    dec_prompt_text = prompts[event_type].strip()
                    assert dec_prompt_text
                    dec_prompt_text = ' '.join(event_name) + ' ' + dec_prompt_text
                    dec_prompt = self.tokenizer(dec_prompt_text, add_special_tokens=True)
                    dec_prompt_ids, dec_prompt_mask_ids = dec_prompt["input_ids"], dec_prompt["attention_mask"]

                    arg_list = self.argument_dict[event_type.replace(':', '.')]
                    arg_2_prompt_slots = dict()
                    arg_2_prompt_slot_spans = dict()
                    num_prompt_slots = 0
                    if os.environ.get("DEBUG", False): arg_set = set()
                    slot_spans_in_event = [] # [(slot_s, slot_e, role_str)]
                    for arg in arg_list:
                        prompt_slots = {
                            "tok_s": list(), "tok_e": list(),
                            "tok_s_off": list(), "tok_e_off": list(),
                        }
                        prompt_slot_spans = []

                        if role_name_mapping is not None:
                            arg_ = role_name_mapping[event_type][arg]
                        else:
                            arg_ = arg
                        # Using this more accurate regular expression might further improve rams results
                        for matching_result in re.finditer(r'\b' + re.escape(arg_) + r'\b',
                                                           dec_prompt_text.split('.')[0]):
                            char_idx_s, char_idx_e = matching_result.span();
                            char_idx_e -= 1
                            tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                            tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                            prompt_slot_spans.append((tok_prompt_s, tok_prompt_e))
                            prompt_slots["tok_s"].append(tok_prompt_s + offset_prompt_);
                            prompt_slots["tok_e"].append(tok_prompt_e + offset_prompt_)
                            prompt_slots["tok_s_off"].append(tok_prompt_s + offset_prompt + offset_prompt_);
                            prompt_slots["tok_e_off"].append(tok_prompt_e + offset_prompt + offset_prompt_)
                            num_prompt_slots += 1
                            assert isinstance(tok_prompt_s, int) and isinstance(tok_prompt_e, int), "char_to_token returned None!"
                            assert 0 <= tok_prompt_s < self.args.max_enc_seq_length
                            assert tok_prompt_e <= self.args.max_enc_seq_length

                        # trigger <-> role slot (intra event)
                        for (tok_prompt_s, tok_prompt_e) in prompt_slot_spans:
                            slot_s = tok_prompt_s + offset_prompt + offset_prompt_
                            slot_e = tok_prompt_e + offset_prompt + offset_prompt_

                            # trigger -> slot
                            enc_attention_mask[0, enc_trigger_start:enc_trigger_end, slot_s:slot_e] = 1
                            # slot -> trigger
                            enc_attention_mask[0, slot_s:slot_e, enc_trigger_start:enc_trigger_end] = 1

                            slot_spans_in_event.append((slot_s, slot_e, arg_))
                            all_event_slots.append((event_type, slot_s, slot_e, arg_))

                        arg_2_prompt_slots[arg] = prompt_slots
                        arg_2_prompt_slot_spans[arg] = prompt_slot_spans
                        
                        # role <-> role slot (intra event)
                        for (s1, e1, r1), (s2, e2, r2) in itertools.combinations(slot_spans_in_event, 2):
                            if (event_type, r1, r2) not in intra_slot_slot.keys():
                                intra_slot_slot[event_type, r1, r2] = 0.0
                            w = intra_slot_slot[event_type, r1, r2]

                            enc_attention_mask[0, s1:e1, s2:e2] = w
                            enc_attention_mask[0, s2:e2, s1:e1] = w
                    

                    list_arg_2_prompt_slots.append(arg_2_prompt_slots)
                    list_num_prompt_slots.append(num_prompt_slots)
                    list_dec_prompt_ids.append(dec_prompt_ids)
                    list_arg_2_prompt_slot_spans.append(arg_2_prompt_slot_spans)
                    

                
                
                offset_prompt_ += len(dec_prompt_ids)
                dec_table_ids += dec_prompt_ids
                dec_table_mask += dec_prompt_mask_ids

            # # inter event 
            for idx1 in range(len(all_event_slots)):
                et1, s1, e1, r1 = all_event_slots[idx1]
                for idx2 in range(idx1 + 1, len(all_event_slots)):
                    et2, s2, e2, r2 = all_event_slots[idx2]
                    if et1 == et2:
                        continue
                    key1 = f"{et1}|{r1}|{et2}|{r2}"
                    # assert 0 <= s1 < e1 <= self.args.max_enc_seq_length
                    # assert 0 <= s2 < e2 <= self.args.max_enc_seq_length
                    if key1 not in param_coref_weight:
                        param_coref_weight[key1] = 0.0
                    
                    w1 = param_coref_weight[key1] 

                    enc_attention_mask[1, s1:e1, s2:e2] = w1
                    enc_attention_mask[1, s2:e2, s1:e1] = w1
                    
            all_ids.extend(dec_table_ids)

            all_mask_ids.extend(dec_table_mask)
            if len(all_ids) > self.args.max_enc_seq_length:
                over_nums += 1

            while len(all_ids) < self.args.max_enc_seq_length:
                all_ids.append(self.tokenizer.pad_token_id)
                all_mask_ids.append(self.args.pad_mask_token)

            row_index = 0
            list_trigger_pos = []
            list_arg_slots = []
            list_target_info = []
            list_roles = []
            """ Deal with target arguments """
            k = 0
            for i, (event_type, events) in enumerate(event_type_2_events.items()):
                for event in events:
                    arg_2_prompt_slots = list_arg_2_prompt_slots[k]
                    num_prompt_slots = list_num_prompt_slots[k]
                    dec_prompt_ids = list_dec_prompt_ids[k]
                    k += 1
                    row_index += 1

                    list_trigger_pos.append(len(dec_table_ids))

                    arg_slots = []
                    cursor = len(dec_table_ids) + 1
                    event_args = event['args']

                    arg_set = set([tuple(arg[:2]) for arg in event_args])

                    event_args_name = [arg[-1] for arg in event_args]
                    target_info = dict()
                    for arg, prompt_slots in arg_2_prompt_slots.items():
                        num_slots = len(prompt_slots['tok_s'])
                        arg_slots.append([cursor + x for x in range(num_slots)])
                        cursor += num_slots

                        arg_target = {"text": list(), "span_s": list(), "span_e": list()}
                        answer_texts, start_positions, end_positions = list(), list(), list()
                        if arg in event_args_name:
                            # Deal with multi-occurance
                            if os.environ.get("DEBUG", False): arg_set.add(arg)
                            arg_idxs = [j for j, x in enumerate(event_args_name) if x == arg]
                            if os.environ.get("DEBUG", False): counter[0] += 1; counter[1] += len(arg_idxs)

                            for arg_idx in arg_idxs:
                                event_arg_info = event_args[arg_idx]
                                answer_text = event_arg_info[2]
                                answer_texts.append(answer_text)
                                start_old, end_old = event_arg_info[0], event_arg_info[1]
                                start_position = old_tok_to_new_tok_index[start_old][0]
                                start_positions.append(start_position)
                                end_position = old_tok_to_new_tok_index[end_old - 1][1]
                                end_positions.append(end_position)

                        arg_target["text"] = answer_texts
                        arg_target["span_s"] = start_positions
                        arg_target["span_e"] = end_positions
                        target_info[arg] = arg_target

                    assert sum([len(slots) for slots in arg_slots]) == num_prompt_slots

                    # dec_table_ids += dec_event_ids
                    list_arg_slots.append(arg_slots)
                    list_target_info.append(target_info)
                    roles = self.argument_dict[event_type.replace(':', '.')]
                    assert len(roles) == len(arg_slots)
                    list_roles.append(roles)

            max_dec_seq_len = self.args.max_prompt_seq_length

            while len(dec_table_ids) < max_dec_seq_len:
                dec_table_ids.append(self.tokenizer.pad_token_id)
                dec_table_mask.append(self.args.pad_mask_token)
            if len(all_ids) > self.args.max_enc_seq_length:
                enc_attention_mask = torch.zeros((2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
                                                 dtype=torch.float32)
            if len(all_ids) > self.args.max_enc_seq_length:
                all_ids = all_ids[:self.args.max_enc_seq_length]
                all_mask_ids = all_mask_ids[:self.args.max_enc_seq_length]

            if len(list_arg_2_prompt_slots) == 1:
                enc_attention_mask = torch.zeros((2, self.args.max_enc_seq_length, self.args.max_enc_seq_length),
                                                 dtype=torch.float32)
                # deal single prompt
                """ Deal with single prompt template """
                offset_prompt_ = 0
                kk = 0
                all_event_slots = []   # [(event_type, slot_s, slot_e, role_str)]
                for i, event_type in enumerate(event_type_2_events):
                    events = event_type_2_events[event_type]
                    event_name = event_type.split('.')
                    event_name = ['<e-%d>' % (i)] + event_name + ['</e-%d>' % (i)]
                    for event in events:
                        enc_trigger_start, enc_trigger_end = trigger_enc_token_index[kk][0] - 1, \
                                                            trigger_enc_token_index[kk][1] + 1

                        s_trig = trigger_enc_token_index[kk][0]     
                        e_trig = trigger_enc_token_index[kk][1]     
                        
                        assert isinstance(s_trig, int) and isinstance(e_trig, int)
                        assert 0 <= s_trig < e_trig <= self.args.max_enc_seq_length, \
                            f"Trigger span 越界: {(s_trig, e_trig)} vs max {self.args.max_enc_seq_length}"

                        kk += 1
                        dec_prompt_text = prompts[event_type].strip()
                        assert dec_prompt_text
                        dec_prompt_text = ' '.join(event_name) + ' ' + dec_prompt_text
                        dec_prompt = self.tokenizer(dec_prompt_text, add_special_tokens=True)
                        dec_prompt_ids, dec_prompt_mask_ids = dec_prompt["input_ids"], dec_prompt["attention_mask"]

                        arg_list = self.argument_dict[event_type.replace(':', '.')]
                        num_prompt_slots = 0
                        if os.environ.get("DEBUG", False): arg_set = set()
                        slot_spans_in_event = [] # [(slot_s, slot_e, role_str)]
                        for arg in arg_list:
                            prompt_slots = {
                                "tok_s": list(), "tok_e": list(),
                                "tok_s_off": list(), "tok_e_off": list(),
                            }
                            prompt_slot_spans = []

                            if role_name_mapping is not None:
                                arg_ = role_name_mapping[event_type][arg]
                            else:
                                arg_ = arg
                            # Using this more accurate regular expression might further improve rams results
                            for matching_result in re.finditer(r'\b' + re.escape(arg_) + r'\b',
                                                            dec_prompt_text.split('.')[0]):
                                char_idx_s, char_idx_e = matching_result.span();
                                char_idx_e -= 1
                                tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                                tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                                prompt_slot_spans.append((tok_prompt_s, tok_prompt_e))
                                prompt_slots["tok_s"].append(tok_prompt_s + offset_prompt_);
                                prompt_slots["tok_e"].append(tok_prompt_e + offset_prompt_)
                                prompt_slots["tok_s_off"].append(tok_prompt_s + offset_prompt + offset_prompt_);
                                prompt_slots["tok_e_off"].append(tok_prompt_e + offset_prompt + offset_prompt_)
                                num_prompt_slots += 1
                                assert isinstance(tok_prompt_s, int) and isinstance(tok_prompt_e, int), "char_to_token returned None!"
                                assert 0 <= tok_prompt_s < self.args.max_enc_seq_length
                                assert tok_prompt_e <= self.args.max_enc_seq_length

                            # trigger <-> role slot (intra event)
                            for (tok_prompt_s, tok_prompt_e) in prompt_slot_spans:
                                slot_s = tok_prompt_s + offset_prompt + offset_prompt_
                                slot_e = tok_prompt_e + offset_prompt + offset_prompt_

                                # trigger -> slot
                                enc_attention_mask[0, enc_trigger_start:enc_trigger_end, slot_s:slot_e] = 1
                                # slot -> trigger
                                enc_attention_mask[0, slot_s:slot_e, enc_trigger_start:enc_trigger_end] = 1

                                slot_spans_in_event.append((slot_s, slot_e, arg_))
                                all_event_slots.append((event_type, slot_s, slot_e, arg_))

                            arg_2_prompt_slots[arg] = prompt_slots
                            
                            # role <-> role slot (intra event)
                            for (s1, e1, r1), (s2, e2, r2) in itertools.combinations(slot_spans_in_event, 2):
                                if (event_type, r1, r2) not in intra_slot_slot.keys():
                                    intra_slot_slot[event_type, r1, r2] = 0.0
                                w = intra_slot_slot[event_type, r1, r2]

                                enc_attention_mask[0, s1:e1, s2:e2] = w
                                enc_attention_mask[0, s2:e2, s1:e1] = w
                # NOTE: one annotation as one decoding input
            feature_idx = len(features)

            features.append(
                InputFeatures(example_id, feature_idx,
                              list_event_type, trigger_enc_token_index,
                              enc_text, enc_input_ids, enc_mask_ids, all_ids, all_mask_ids,
                              dec_prompt_text, dec_table_ids, dec_table_mask, None,
                              list_arg_2_prompt_slots, list_target_info, enc_attention_mask,
                              old_tok_to_new_tok_index=old_tok_to_new_tok_index, full_text=example.context,
                              arg_list=list_roles
                              )
            )
            # enc_tokens = self.tokenizer.convert_ids_to_tokens(enc_input_ids)
            # enc_tokens_ = dict()
            # for i, token in enumerate(enc_tokens):
            #     enc_tokens_[i] = token
            # print(enc_tokens_)

            # enc_tokens = self.tokenizer.convert_ids_to_tokens(all_ids)
            # enc_tokens_ = dict()
            # for i, token in enumerate(enc_tokens):
            #     enc_tokens_[i] = token
            # print(enc_tokens_)

            # enc_tokens = self.tokenizer.convert_ids_to_tokens(all_ids)
            # enc_tokens_ = dict()
            # for i, token in enumerate(enc_tokens):
            #     enc_tokens_[i] = token
            # print(enc_tokens_)



        # print(over_nums)
        if os.environ.get("DEBUG", False): print(
            '\033[91m' + f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})" + '\033[0m')
        return features

    def convert_features_to_dataset(self, features):
        dataset = ArgumentExtractionDataset(features)
        return dataset