project('par_comp_lab1', 'cpp', default_options: ['cpp_std=c++23', 'buildtype=release', 'b_ndebug=if-release'])
compiler = meson.get_compiler('cpp')
compiler_version = compiler.version()

if compiler.get_id() == 'gcc' and compiler_version.version_compare('<14.0')
  error('Require GCC version >=14.0')
elif compiler.get_id() == 'clang' and compiler_version.version_compare('<18.0')
  error('Require clang version >=18.0')
endif
add_project_link_arguments('-Wl,--copy-dt-needed-entries', language: 'cpp')
#add_project_link_arguments('-flto', language: 'cpp')
#add_project_link_arguments('-fuse-ld=gold', language: 'cpp')

cmake = import('cmake')
# get matplot++ library
matplotplusplus = cmake.subproject('matplotplusplus')
matplot_dep = matplotplusplus.dependency('matplot')
nodesoup_dep = matplotplusplus.dependency('nodesoup')

boost_dep = dependency('boost', modules: ['mpi'])
executable('calc_pi', 'calc_pi.cc', dependencies: boost_dep)
executable('comm_time', 'comm_time.cc', dependencies: boost_dep)
executable('conv_diff', 'conv_diff.cc', dependencies: [matplot_dep, nodesoup_dep, boost_dep], install: true)
