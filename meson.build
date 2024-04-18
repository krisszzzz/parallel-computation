
project('par_comp_lab1', 'cpp', default_options: ['cpp_std=c++23', 'optimization=3'])
compiler = meson.get_compiler('cpp')
compiler_version = compiler.version()

if compiler.get_id() == 'gcc' and compiler_version.version_compare('<14.0')
  error('Require GCC version >=14.0')
elif compiler.get_id() == 'clang' and compiler_version.version_compare('<18.0')
  error('Require clang version >=18.0')
endif
add_project_link_arguments('-Wl,--copy-dt-needed-entries', language: 'cpp')

boost_dep = dependency('boost', modules: ['mpi'])
executable('calc_pi', 'calc_pi.cc', dependencies: boost_dep)
executable('comm_time', 'comm_time.cc', dependencies: boost_dep)