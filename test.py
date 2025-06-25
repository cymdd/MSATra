from config import get_parser
from date_loader import Data_loader
from main import os,Main
from utils import load_arg,save_arg
import torch

def train(args,parser):
    domains = ['eth', 'hotel', 'zara01', 'zara02', 'students001', 'students003', 'uni_examples', 'zara03']
    #根据domains选择源域和目标域下标:
    # 如 source_domain = 0   ->  源域：eth
    #   target_domain = 1   -> 目标域: hotel
    source_domain = 6
    target_domain = 3
    train_set = [source_domain, target_domain]
    #-----创建存储文件夹-----
    model_dirName = f'{domains[train_set[0]]}2{domains[train_set[1]]}'
    args.model_filepath = os.path.join(args.checkpoint_dir, model_dirName)
    if not os.path.exists(args.model_filepath):
        # 创建文件夹
        os.makedirs(args.model_filepath)
        print(f"文件夹 '{model_dirName}' 已创建在 '{args.checkpoint_dir}' 中。")
    #-----加载/创建配置文件----
    args.phase = 'test'
    args.config = args.model_filepath+'/config_'+args.phase+'.yaml'
    if not load_arg(args,parser):
        save_arg(args)
    args = load_arg(args,parser)
    Dataloader = Data_loader(args, train_set, phase="test")
    main = Main(args, Dataloader)
    main.playtest()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.using_cuda:
        torch.cuda.set_device(args.gpu)
    checkpoint_dir = "./checkpoints/"
    args.checkpoint_dir = checkpoint_dir
    if os.path.exists(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)
    train(args,parser)