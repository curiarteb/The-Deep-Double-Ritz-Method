import pandas as pd, numpy as np, tensorflow as tf

def save_evaluation(x, func, name, file_name):
    out = func(x)
    x_out = tf.concat([x, out], axis=1)
    data = pd.DataFrame(x_out, columns=["x", name])
    data.index.name = "sample"
    data.to_csv(file_name, index=False, sep="\t")
    return out.numpy().squeeze()

def save_loss(loss, file_name):
    data = pd.DataFrame(loss, columns=["loss_uv"])
    data.index.name = "iteration"
    data.to_csv(file_name, index=True, sep="\t")