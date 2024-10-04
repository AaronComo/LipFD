import os
import argparse
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14", help="see models/__init__.py")
        parser.add_argument("--fix_backbone", default=False)
        parser.add_argument("--fix_encoder", default=True)

        parser.add_argument("--real_list_path", default="./datasets/val/0_real")
        parser.add_argument("--fake_list_path", default="./datasets/val/1_fake")
        parser.add_argument("--data_label", default="train", help="label to decide whether train or validation dataset",)

        parser.add_argument( "--batch_size", type=int, default=10, help="input batch size")
        parser.add_argument("--gpu_ids", type=str, default="1", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",)
        parser.add_argument("--name", type=str, default="experiment_name", help="name of the experiment. It decides where to store samples and models",)
        parser.add_argument("--num_threads", default=0, type=int, help="# threads for loading data")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here",)
        parser.add_argument("--serial_batches",action="store_true",help="if true, takes images in order to make batches, otherwise takes them randomly",)
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        # util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # additional
        # opt.classes = opt.classes.split(',')
        opt.rz_interp = opt.rz_interp.split(",")
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(",")]
        opt.jpg_method = opt.jpg_method.split(",")
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(",")]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt
