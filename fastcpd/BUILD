load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")

py_library(
    name = "fastcpd_init",
    srcs = [
        "__init__.py",
        "segmentation.py",
        "variance_estimation.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":py_fastcpd",
    ],
)

py_library(
    name = "fastcpd_py_library",
    visibility = ["//visibility:public"],
    deps = [
        ":fastcpd_init",
        ":py_fastcpd",
    ],
)

py_package(
    name = "fastcpd_py_package",
    visibility = ["//visibility:public"],
    deps = [
        ":fastcpd_py_library",
    ],
)

py_wheel(
    name = "pyfastcpd",
    # Package data. We're building "example_minimal_package-0.0.1-py3-none-any.whl"
    distribution = "fastcpd",
    python_tag = "py3",
    stamp = 0,
    version = "0.9.0",
    deps = [
        ":fastcpd_py_package",
    ],
)

cc_binary(
    name = "interface",
    srcs = ["pybind.cc"],
    copts = ["-std=c++17"],
    includes = [
        ".",
    ],
    linkopts = select({
        "@platforms//os:osx": ["-undefined dynamic_lookup"],
        "@platforms//os:linux": [],
        "@platforms//os:windows": [],
    }),
    linkshared = True,
    linkstatic = select({
        "@platforms//os:osx": False,
        "@platforms//os:linux": True,
        "@platforms//os:windows": True,
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//cc/cclib:class",
        "@com_github_pybind_pybind11//:pybind11",
        "@local_config_python//:python_headers",
    ],
)

# This wrapper exists to expose the .so binary to python
py_library(
    name = "py_fastcpd",
    data = [
        ":interface",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
)
