load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@emz_deps//:requirements.bzl", "requirement")

py_library(
    name = "pointnet_model",
    srcs = ["pointnet_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("jax"),
        requirement("jaxlib"),
        requirement("dm_haiku"),
    ],
)

py_test(
    name = "pointnet_model_test",
    srcs = ["pointnet_model_test.py"],
    deps = [
        ":pointnet_model",
        requirement("jax"),
        requirement("dm_haiku"),
    ],
)
