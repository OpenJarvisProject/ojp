# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/media/ojp/vendor/llama-crab-sys/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders")
  file(MAKE_DIRECTORY "/media/ojp/vendor/llama-crab-sys/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders")
endif()
file(MAKE_DIRECTORY
  "/media/ojp/target-vk/release/build/llama-crab-sys-75c85abe0e909ed7/out/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build"
  "/media/ojp/target-vk/release/build/llama-crab-sys-75c85abe0e909ed7/out/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix"
  "/media/ojp/target-vk/release/build/llama-crab-sys-75c85abe0e909ed7/out/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/tmp"
  "/media/ojp/target-vk/release/build/llama-crab-sys-75c85abe0e909ed7/out/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp"
  "/media/ojp/target-vk/release/build/llama-crab-sys-75c85abe0e909ed7/out/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src"
  "/media/ojp/target-vk/release/build/llama-crab-sys-75c85abe0e909ed7/out/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/media/ojp/target-vk/release/build/llama-crab-sys-75c85abe0e909ed7/out/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/media/ojp/target-vk/release/build/llama-crab-sys-75c85abe0e909ed7/out/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp${cfgdir}") # cfgdir has leading slash
endif()
