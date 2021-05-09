load("@rules_python//python:defs.bzl", "py_library")
load("@emz_deps//:requirements.bzl", "requirement")

py_library(
    name = "orbit",
    srcs = glob(["orbit/**/*.py"]),
    visibility = ["//visibility:public"],
    deps = [
        requirement("tensorflow"),
        requirement("absl_py"),
    ],
)

py_library(
    name = "hyperparams",
    srcs = [
        "official/modeling/hyperparams/base_config.py",
        "official/modeling/hyperparams/params_dict.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        requirement("pyyaml"),
        requirement("tensorflow"),
        requirement("absl_py"),
    ],
)

py_library(
    name = "performance",
    srcs = ["official/modeling/performance.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("tensorflow"),
        requirement("absl_py"),
    ],
)

py_library(
    name = "nlp_optimization",
    srcs = ["official/nlp/optimization.py"],
    deps = [
        requirement("absl_py"),
        requirement("gin_config"),
        requirement("tensorflow"),
        requirement("tensorflow_addons"),
    ],
)

py_library(
    name = "optimization",
    srcs = glob(["official/modeling/optimization/**/*.py"]),
    visibility = ["//visibility:public"],
    deps = [
        ":nlp_optimization",
        requirement("absl_py"),
        requirement("gin_config"),
        requirement("tensorflow"),
        requirement("tensorflow_addons"),
    ],
)

py_library(
    name = "core",
    srcs = glob(["official/core/*.py"]),
    visibility = ["//visibility:public"],
    deps = [
        ":optimization",
        ":performance",
        requirement("absl_py"),
        requirement("tensorflow"),
    ],
)
