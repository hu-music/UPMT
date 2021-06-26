from model import PopMusicTransformer
import argparse
import tensorflow as tf
import os
import pickle
import numpy as np
from glob import glob
parser = argparse.ArgumentParser(description='')
parser.add_argument('--prompt_path', dest='prompt_path', default='./test/prompt/test_input.mid', help='path of input')
parser.add_argument('--output_path', dest='output_path', default='./test/g/test_generate.mid', help='path of the output')
parser.add_argument('--favorite_path', dest='favorite_path', default='./test/favorite/test_favorite.mid', help='path of favorite')
parser.add_argument('--trainingdata_path', dest='trainingdata_path', default='./test/data/training.pickle', help='path of favorite training data')
parser.add_argument('--output_checkpoint_folder', dest='output_checkpoint_folder', default='./test/checkpoint/', help='path of checkpoint')
parser.add_argument('--alpha', default=0.1, help='weight of events')
parser.add_argument('--temperature', default=300, help='sampling temperature')
parser.add_argument('--topk', default=5, help='sampling topk')
parser.add_argument('-l','--smpi', nargs='+',default=[-2,-2,-1,-2,-2,2,2,5], help='signature music pattern interval')
parser.add_argument('--type', dest='type', default='generateno', help='generateno or pretrain or prepare')
args = parser.parse_args()


def main(_):

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tfconfig) as sess:


        if args.type == 'prepare':
            midi_paths = glob('./test/favorite'+'/*.mid')
            model = PopMusicTransformer(
                checkpoint='./test/model',
                is_training=False)
            model.prepare_data(
                        midi_paths=midi_paths)    
        
        
        elif args.type =='pretrain':
            training_data = pickle.load(open(args.trainingdata_path,"rb"))
            if not os.path.exists(args.output_checkpoint_folder):
                os.mkdir(args.output_checkpoint_folder)
            model = PopMusicTransformer(
                checkpoint='./test/model',
                is_training=True)
            model.finetune(
                training_data=training_data,
                alpha=float(args.alpha),
                favoritepath=args.favorite_path,
                output_checkpoint_folder=args.output_checkpoint_folder)
            
            
        elif args.type == 'generateno':
            model = PopMusicTransformer(
                checkpoint='./test/model',
                is_training=False)
            model.generate_noteon(
                        temperature=float(args.temperature),
                        topk=int(args.topk),
                        output_path=args.output_path,  
                        smpi= np.array(args.smpi),
                        prompt=args.prompt_path)


if __name__ == '__main__':
    tf.app.run()
