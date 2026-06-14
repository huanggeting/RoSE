import json, glob, os
from collections import defaultdict
from tqdm import tqdm
import jsonlines


def build_ace_coref_weight(train_glob,
                           save_path="./data/dset_meta/role_coref_weight_ace.json",
                           smooth=1):
    """
    train_glob : '*.json' 匹配 ACE train split 的所有文件
    β = (c_ij + smooth)/(c_i + smooth)
    role_key = f'{event_type}|{role}'
    """
    PAIR_CNT = defaultdict(int)    
    ROLE_CNT = defaultdict(int)
    for fp in tqdm(glob.glob(train_glob), desc="[ACE coref stat]"):
        with open(fp) as f:
            for line in f:
                doc = json.loads(line)

                
                ev_args = []
                if doc["event_mentions"]:
                    for item in doc["event_mentions"]:
                        etype = item["event_type"].replace(":", ".")
                        lst = []
                        for arg in item["arguments"]:
                            role_key = f"{etype}|{arg['role']}"
                            ent_id   = arg["entity_id"]
                            lst.append((role_key, ent_id))
                            ROLE_CNT[role_key] += 1
                        ev_args.append(lst)

                
                for idx_a in range(len(ev_args)):
                    for idx_b in range(idx_a+1, len(ev_args)):
                        for rk1, ent1 in ev_args[idx_a]:
                            for rk2, ent2 in ev_args[idx_b]:
                                if ent1 == ent2:
                                    PAIR_CNT[(rk1, rk2)] += 1
                                    PAIR_CNT[(rk2, rk1)] += 1

    weight = {}
    for (rk1, rk2), c_ij in PAIR_CNT.items():
        c_i  = ROLE_CNT[rk1]
        beta = (c_ij + smooth) / (c_i + smooth)
        weight[f"{rk1}|{rk2}"] = round(beta, 4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(weight, open(save_path, "w"), indent=2)
    print(f"[save] {len(weight)} β weights → {save_path}")
    return weight


def parse_a1(a1_path):
    entity_dict = {}
    with open(a1_path, encoding = 'utf-8') as f:
        for line in f:
            if line.startswith("T"):
                parts = line.strip().split('\t')
                ent_id = parts[0]
                rest = parts[1].split()
                ent_type = rest[0]
                span = list(map(int, rest[1:3]))
                text = parts[2]
                entity_dict[ent_id] = {
                    "type": ent_type,
                    "span": span,
                    "text": text
                }
    return entity_dict


def parse_a2(a2_path):
    event_list = []
    with open(a2_path, encoding='utf-8') as f:
        for line in f:
            if line.startswith("E"):
                fields = line.strip().split("\t")
                if len(fields) < 2:
                    continue
                rest = fields[1]
                args = rest.split()
                if not args:
                    continue
                trigger_type, trigger_eid = args[0].split(":")
                event_params = []
                event_params.append((trigger_type, "Trigger", trigger_eid))
                for arg in args[1:]:
                    if ":" in arg:
                        role, eid = arg.split(":")
                        event_params.append((trigger_type, role, eid))
                event_list.append(event_params)
    return event_list


def build_mlee_coref_weight(folder, a1_glob='*.a1', a2_glob='*.a2', save_path='./data/dset_meta/role_coref_weight_mlee.json', smooth=1):
    PAIR_CNT = defaultdict(int)
    ROLE_CNT = defaultdict(int)
    
    a1_files = sorted(glob.glob(os.path.join(folder, "*.a1")))
    for a1_path in a1_files:
        a2_path = a1_path[:-3] + ".a2"
        if not os.path.exists(a2_path):
            continue
        entity_dict = parse_a1(a1_path)
        event_list = parse_a2(a2_path)
        
        for event_params in event_list:
            for (etype, role, eid) in event_params:
                key = f"{etype}|{role}"
                ROLE_CNT[key] += 1
        
        for idx1, args1 in enumerate(event_list):
            for idx2, args2 in enumerate(event_list):
                if idx1 >= idx2:continue
                for (etype1, role1, eid1) in args1:
                    for (etype2, role2, eid2) in args2:
                        if eid1 == eid2:
                            PAIR_CNT[(f"{etype1}|{role1}", f"{etype2}|{role2}")] += 1
                            PAIR_CNT[(f"{etype2}|{role2}", f"{etype1}|{role1}")] += 1

    weight = {}
    for (rk1, rk2), c_ij in PAIR_CNT.items():
        c_i  = ROLE_CNT[rk1]
        beta = (c_ij + smooth) / (c_i + smooth)                 
        weight[f"{rk1}|{rk2}"] = round(beta, 4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(weight, open(save_path, "w"), indent=2)
    print(f"[save] {len(weight)} β weights → {save_path}")
    return weight


def build_wikievent_coref_weight(json_path, save_path = "./data/dset_meta/role_coref_weight_wikievent.json", smooth = 1):
    PAIR_CNT = defaultdict(int)
    ROLE_CNT = defaultdict(int)
    with open(json_path, encoding = 'utf-8') as f:
        args = []
        for line in f:
            data = json.loads(line)
            for evt in data["event_mentions"]:
                evt_type = evt["event_type"]
                for arg in evt.get("arguments", []):
                    role = arg["role"]
                    ent_id = arg["entity_id"]
                    text = arg["text"]
                    args.append((evt_type, role, ent_id, text))
                    print (args[0])
                    ROLE_CNT[f"{evt_type}|{role}"] += 1

        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                etype1, role1, eid1, text1 = args[i]
                etype2, role2, eid2, text2 = args[j]
                if eid1 == eid2:
                    PAIR_CNT[(f"{etype1}|{role1}", f"{etype2}|{role2}")] += 1
                    PAIR_CNT[(f"{etype2}|{role2}", f"{etype1}|{role1}")] += 1

    weight = {}
    for (rk1, rk2), c_ij in PAIR_CNT.items():
        c_i  = ROLE_CNT[rk1]
        beta = (c_ij + smooth) / (c_i + smooth)                 
        weight[f"{rk1}|{rk2}"] = round(beta, 4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(weight, open(save_path, "w"), indent=2)
    print(f"[save] {len(weight)} β weights → {save_path}")
    return weight


def build_rams_coref_weight(json_path, save_path = "./data/dset_meta/role_coref_weight_rams.json", smooth = 1):
    PAIR_CNT = defaultdict(int)
    ROLE_CNT = defaultdict(int)
    lines = []
    with jsonlines.open(json_path) as reader:
        for obj in reader:
            lines.append(obj)

    for line in lines:
        args = []
        for evt in line["events"]:
            evt_type = evt["event_type"]
            for arg in evt["args"]:
                role = arg[3]
                ent_span = (arg[0], arg[1])
                args.append((evt_type, role, ent_span))
                ROLE_CNT[f"{evt_type}|{role}"] += 1

        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                etype1, role1, ent_span1 = args[i]
                etype2, role2, ent_span2 = args[j]
                if ent_span1 == ent_span2:
                    PAIR_CNT[(f"{etype1}|{role1}", f"{etype2}|{role2}")] += 1
                    PAIR_CNT[(f"{etype2}|{role2}", f"{etype1}|{role1}")] += 1

    weight = {}
    for (rk1, rk2), c_ij in PAIR_CNT.items():
        c_i  = ROLE_CNT[rk1]
        beta = (c_ij + smooth) / (c_i + smooth)                
        weight[f"{rk1}|{rk2}"] = round(beta, 4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(weight, open(save_path, "w"), indent=2)
    print(f"[save] {len(weight)} β weights → {save_path}")
    return weight

if __name__ == "__main__":
    build_ace_coref_weight("data/ace_eeqa/oneie_data/train.w1.oneie.json")
    folder = "./data/MLEE/raw/train"
    build_mlee_coref_weight(folder, save_path = "./data/dset_meta/role_coref_weight_mlee.json")

    build_wikievent_coref_weight(json_path = "data/WikiEvent/data/train.jsonl", save_path = "./data/dset_meta/role_coref_weight_wikievent.json")
    build_rams_coref_weight(json_path = "data/RAMS_1.0/data_final/train.jsonlines", save_path = "./data/dset_meta/role_coref_weight_rams.json")


