load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@emz_deps//:requirements.bzl", "requirement")

py_library(
    name = "nn_blocks",
    srcs = ["nn_blocks.py"],
    visibility = ["//visibility:public"],
    deps = [requirement("tensorflow")],
)

py_test(
    name = "nn_blocks_test",
    srcs = ["nn_blocks_test.py"],
    deps = [
        ":nn_blocks",
        requirement("tensorflow"),
    ],
)
