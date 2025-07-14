import random
import argparse
import os
import pickle as pkl

def main(args):
    if not os.path.exists(args.reward_datasets_dir):
         os.makedirs(args.reward_datasets_dir)
    with open(args.demos_path, "rb") as f:
        transitions = pkl.load(f)
    random.shuffle(transitions)
    dev_ratio = 0.2
    dev_transitions = transitions[:int(dev_ratio * len(transitions))]
    
    with open(args.reward_datasets_dir + '/dev.pkl', "wb") as f:
        pos_transitions = list(filter(lambda x: x['rewards'], dev_transitions))
        neg_transitions = list(filter(lambda x: not x['rewards'], dev_transitions))
        print(f'dev: pos: {len(pos_transitions)}, neg: {len(neg_transitions)}')
        pkl.dump(dev_transitions, f)
    
    transitions = transitions[int(dev_ratio * len(transitions)):]
    pos_transitions = list(filter(lambda x: x['rewards'], transitions))
    neg_transitions = list(filter(lambda x: not x['rewards'], transitions))
    
    print(f'pos: {len(pos_transitions)}, neg: {len(neg_transitions)}')
    
    with open(args.reward_datasets_dir + '/success.pkl', "wb") as f:
        pkl.dump(pos_transitions, f)
        
    with open(args.reward_datasets_dir + '/failure.pkl', "wb") as f:
        pkl.dump(neg_transitions, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demos_path', default='tasks/take_cup/init/demos/2025-02-10_16-23-51/rlpd_demos.pickle', type=str, required=False, help='load dataset directory')
    parser.add_argument('--reward_datasets_dir', default='tasks/take_cup/classifier_data', type=str, required=False, help='save dataset directory')
    main(parser.parse_args())
