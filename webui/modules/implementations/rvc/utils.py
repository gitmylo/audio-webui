import os
import traceback
from collections import OrderedDict

import torch


def savee(ckpt, sr, if_f0, name, epoch, version):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        if sr == "40k":
            opt["config"] = [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2],
                512,
                [16, 16, 4, 4],
                109,
                256,
                40000,
            ]
        elif sr == "48k":
            opt["config"] = [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 6, 2, 2, 2],
                512,
                [16, 16, 4, 4, 4],
                109,
                256,
                48000,
            ]
        elif sr == "32k":
            opt["config"] = [
                513,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 4, 2, 2, 2],
                512,
                [16, 16, 4, 4, 4],
                109,
                256,
                32000,
            ]
        opt["info"] = "%sepoch" % epoch
        opt["sr"] = sr
        opt["f0"] = if_f0
        opt["version"] = version
        os.makedirs(os.path.dirname(name), exist_ok=True)
        torch.save(opt, name)
        return "Success."
    except:
        return traceback.format_exc()


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # 模型需要的shape
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                print(
                    "shape-%s-mismatch|need-%s|get-%s"
                    % (k, state_dict[k].shape, saved_state_dict[k].shape)
                )  #
                raise KeyError
        except:
            # logger.info(traceback.format_exc())
            new_state_dict[k] = v  # 模型自带的随机值
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
    ):  ###加载不了，如果是空的的话，重新初始化，可能还会影响lr时间表的更新，因此在train文件最外围catch
        #   try:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    #   except:
    #     traceback.print_exc()
    return model, optimizer, learning_rate, iteration