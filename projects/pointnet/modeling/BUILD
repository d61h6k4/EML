load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@emz_deps//:requirements.bzl", "requirement")

py_library(
    name = "pointnet",
    srcs = ["pointnet.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("jax"),
        requirement("jaxlib"),
        requirement("dm_haiku"),
    ],
)

py_test(
    name = "pointnet_test",
    srcs = ["pointnet_test.py"],
    deps = [
        ":pointnet",
        requirement("jax"),
        requirement("dm_haiku"),
    ],
)
