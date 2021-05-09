workspace(name = "emz")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    sha256 = "778197e26c5fbeb07ac2a2c5ae405b30f6cb7ad1f5510ea6fdac03bded96cc6f",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.2.0/rules_python-0.2.0.tar.gz",
)

load("@rules_python//python:pip.bzl", "pip_install")

pip_install(
    name = "emz_deps",
    requirements = "//:requirements.txt",
)

_TENSORFLOW_MODELS_VERSION = "7f0ee4cb1f10d4ada340cc5bfe2b99d0d690b219"

http_archive(
    name = "tensorflow_models",
    build_file = "//third_party:tensorflow_models.BUILD",
    sha256 = "2d85a7d680fc918cf02f191b2093a754dcaa4fcc5cdd39a6b4d95eb6fb7f81ee",
    strip_prefix = "models-{}".format(_TENSORFLOW_MODELS_VERSION),
    urls = ["https://github.com/tensorflow/models/archive/{}.zip".format(_TENSORFLOW_MODELS_VERSION)],
)
