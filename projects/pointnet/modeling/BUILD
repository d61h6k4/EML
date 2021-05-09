load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@emz_deps//:requirements.bzl", "requirement")

py_library(
    name = "transformnet_model",
    srcs = ["transformnet_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//projects/pointnet/modeling/layers:nn_blocks",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "pointnet_model",
    srcs = ["pointnet_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//projects/pointnet/modeling/layers:nn_blocks",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "transformnet_model_test",
    srcs = ["transformnet_model_test.py"],
    deps = [
        ":transformnet_model",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "pointnet_model_test",
    srcs = ["pointnet_model_test.py"],
    deps = [
        ":pointnet_model",
        ":transformnet_model",
        requirement("tensorflow"),
    ],
)
