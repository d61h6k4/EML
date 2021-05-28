load("@rules_python//python:defs.bzl", "py_binary")
load("@emz_deps//:requirements.bzl", "requirement")

py_binary(
    name = "train",
    srcs = ["train.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//datasets:modelnet",
        "//projects/pointnet/modeling:pointnet_model",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("dm_haiku"),
        requirement("optax"),
        requirement("tensorflow"),
        requirement("tensorflow_datasets"),
        requirement("tensorboard"),
        requirement("rich"),
        requirement("dm_tree"),
    ],
)
