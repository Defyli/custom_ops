cmake_minimum_required(VERSION 3.18)
project(moe_offload_cpp LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
# Offline configuration uses LibTorch's standard package discovery.
# Only the deployed executable is required to run without Python.
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

add_executable(fuse_moe_offload_aoti ../fuse_moe_offload_aoti.cpp)
set_target_properties(fuse_moe_offload_aoti PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(fuse_moe_offload_aoti PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/../../csrc/fuse_moe")
target_include_directories(fuse_moe_offload_aoti SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
separate_arguments(_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(fuse_moe_offload_aoti PRIVATE ${_torch_cxx_flags})

# Prefer official imported targets; otherwise locate CUDA libraries beside Torch.
foreach(_library IN ITEMS torch_cuda c10_cuda)
  if(TARGET ${_library})
    list(APPEND _torch_cuda_libraries ${_library})
  else()
    find_library(${_library}_EXAMPLE_LIBRARY NAMES ${_library}
      HINTS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
    list(APPEND _torch_cuda_libraries "${${_library}_EXAMPLE_LIBRARY}")
  endif()
endforeach()
target_link_libraries(fuse_moe_offload_aoti PRIVATE
  ${TORCH_LIBRARIES} ${_torch_cuda_libraries} CUDA::cudart
  Threads::Threads ${CMAKE_DL_LIBS})

# Optional symbol provider for a pybind-built native extension, not an interpreter.
# Pass an absolute shared libpython path only if dlopen reports unresolved Py*.
# No direct torch_python link or interpreter initialization in this example.
# LibTorch's upstream offline configuration may discover a Python interpreter.
set(MOE_OFFLOAD_LIBPYTHON "" CACHE FILEPATH
  "Optional shared libpython providing symbols needed by the native extension")
if(MOE_OFFLOAD_LIBPYTHON)
  if(NOT IS_ABSOLUTE "${MOE_OFFLOAD_LIBPYTHON}" OR
      NOT EXISTS "${MOE_OFFLOAD_LIBPYTHON}")
    message(FATAL_ERROR "MOE_OFFLOAD_LIBPYTHON must name an existing absolute shared library")
  endif()
  # Retain this dlopen-only dependency even with the linker's --as-needed default.
  target_link_libraries(fuse_moe_offload_aoti PRIVATE
    "-Wl,--no-as-needed" "${MOE_OFFLOAD_LIBPYTHON}" "-Wl,--as-needed")
endif()
