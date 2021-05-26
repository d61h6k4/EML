load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@emz_deps//:requirements.bzl", "requirement")

py_library(
    name = "modelnet",
    srcs = ["modelnet.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("tensorflow"),
        requirement("tensorflow_datasets"),
        requirement("trimesh"),
    ],
)

py_test(
    name = "modelnet_test",
    srcs = ["modelnet_test.py"],
    data = [
        ":checksums",
        "//datasets/dummy_data:modelnet",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":modelnet",
        requirement("tensorflow"),
        requirement("tensorflow_datasets"),
    ],
)

filegroup(
    name = "checksums",
    srcs = glob(["checksums.tsv"]),
    visibility = ["//datasets:__subpackages__"],
)
