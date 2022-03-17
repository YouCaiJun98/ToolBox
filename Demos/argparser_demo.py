import argparse

def get_parser():
    print("April Fool!")
    parser = argparse.ArgumentParser(description="April Fool - A PyTorch BNN implementation.", prog="main.py")
    #parser.add_argument("cfg", help="Available models: ")
    #parser.add_argument("--local_rank",default=-1,type=int,help="the rank of this process")
    parser.add_argument("--gpus", default="0", help="gpu ids, seperate by comma")
    parser.add_argument("--save-path",  default="./logs", type=str,
                        help="store logs/checkpoints under this directory")
    parser.add_argument("--resume", "-r", help="resume from checkpoint")
    parser.add_argument("--pretrain", action="store_true",
                        help="used with `--resume`, regard as a pretrain model, do not recover epoch/best_acc")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="do not use gpu")
    parser.add_argument("--seed", default=2021, help="random seed", type=int)
    parser.add_argument("--save-every", default=50, type=int, help="save every N epoch")
    #parser.add_argument("--distributed", action="store_true",help="whether to use distributed training")
    parser.add_argument("--dataset", default="cifar", type=str, help="training dataset")
    parser.add_argument("--dataset-path", default="./datasets/cifar10", type=str, help="dataset path")
    return parser
