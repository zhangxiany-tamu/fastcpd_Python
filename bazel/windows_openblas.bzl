def _maybe_openblas_impl(ctx):
    # ctx.os is one of "windows", "linux", "darwin"
    if ctx.os == "windows":
        return ctx.new_local_repository(
            name = ctx.name,
            path = "C:/tools/OpenBLAS",   # adjust to wherever you expand the ZIP
            build_file_content = """
cc_library(
  name     = "openblas",
  hdrs     = glob(['include/**/*.h']),
  linkopts = ['/LIBPATH:C:/tools/OpenBLAS/lib', 'libopenblas.lib'],
  visibility = ['//visibility:public'],
)
""",
        )
    # on non‐Windows hosts, do nothing (repo doesn’t get materialized)
    return []

maybe_openblas = repository_rule(
    implementation = _maybe_openblas_impl,
    local          = True,
)
