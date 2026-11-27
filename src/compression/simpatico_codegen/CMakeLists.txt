cmake_minimum_required(VERSION 3.24)

project(simpatico_codegen LANGUAGES C CXX CUDA)

# libcudf exports its CCCL dependency, including include directories and
# namespace/feature definitions. Use the same targets in every build layout.
if(NOT TARGET cudf::cudf)
  find_package(cudf REQUIRED CONFIG)
endif()
find_package(CUDAToolkit REQUIRED)
if(NOT TARGET nvcomp::nvcomp)
  find_package(nvcomp REQUIRED CONFIG)
endif()

include(cmake/jit_header_manifest.cmake)
include(cmake/cache_identity.cmake)

# Host-side OnPair encoder for the `onpair` string operator (Rust, via cargo +
# Corrosion).
option(SIMPATICO_ONPAIR
       "Build the onpair string operator and its Rust encoder bridge" ON)
option(SIMPATICO_LARGE_MEMORY_TESTS
       "Register opt-in GPU tests requiring multiple GiB" OFF)
if(SIMPATICO_ONPAIR)
  include(cmake/onpair_bridge.cmake)
endif()

# Embed the project headers the runtime NVRTC JIT #includes as raw-string
# literals, so a binary distribution needs no simpatico source tree on disk.
# Regenerated whenever either header or the generator script changes.
set(EMBED_JIT_HEADERS_OUT
    "${CMAKE_CURRENT_BINARY_DIR}/generated/codegen/jit/embedded_headers.h")
add_custom_command(
  OUTPUT "${EMBED_JIT_HEADERS_OUT}"
  COMMAND
    ${CMAKE_COMMAND}
    "-DIN_STDINT=${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/stdint_shim.hpp"
    "-DIN_RLE=${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/decode/rle_block.cuh"
    "-DOUT=${EMBED_JIT_HEADERS_OUT}" "-DXXHSUM_EXECUTABLE=${XXHSUM_EXECUTABLE}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_jit_headers.cmake"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/stdint_shim.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/decode/rle_block.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_jit_headers.cmake"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jit_header_manifest.cmake"
          "${XXHSUM_EXECUTABLE}"
  COMMENT "Embedding JIT project headers into ${EMBED_JIT_HEADERS_OUT}"
  VERBATIM)
add_custom_target(simpatico_embedded_headers DEPENDS "${EMBED_JIT_HEADERS_OUT}")

# Embed the complete CCCL library bundle as named in-memory headers. Runtime
# compilation needs neither an installed header tree nor include-closure rules.
set(EMBED_CCCL_HEADERS_OUT
    "${CMAKE_CURRENT_BINARY_DIR}/generated/codegen/jit/cccl_embedded_headers.cpp"
)
set(EMBED_CCCL_HEADERS_DEPFILE "${EMBED_CCCL_HEADERS_OUT}.d")
# CCCL::CCCL's transitive include directories are resolved at generate time; the
# cmake -P embedder cannot evaluate their generator expressions itself. Pass the
# ordered list to support both installed packages and source checkouts with
# separate cub/, thrust/, and libcudacxx/ roots.
set(EMBED_CCCL_INCLUDE_DIRS
    "${CMAKE_CURRENT_BINARY_DIR}/generated/cccl_include_dirs-$<CONFIG>.txt")
file(
  GENERATE
  OUTPUT "${EMBED_CCCL_INCLUDE_DIRS}"
  CONTENT
    "$<JOIN:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:CCCL::CCCL,INTERFACE_INCLUDE_DIRECTORIES>>,\n>\n"
)
add_custom_command(
  OUTPUT "${EMBED_CCCL_HEADERS_OUT}"
  COMMAND
    ${CMAKE_COMMAND} "-DINCLUDE_DIRS_FILE=${EMBED_CCCL_INCLUDE_DIRS}"
    "-DOUT=${EMBED_CCCL_HEADERS_OUT}" "-DDEPFILE=${EMBED_CCCL_HEADERS_DEPFILE}"
    "-DXXHSUM_EXECUTABLE=${XXHSUM_EXECUTABLE}" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cccl_headers.cmake"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cccl_headers.cmake"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jit_header_manifest.cmake"
          "${EMBED_CCCL_INCLUDE_DIRS}" "${XXHSUM_EXECUTABLE}"
  DEPFILE "${EMBED_CCCL_HEADERS_DEPFILE}"
  COMMENT "Embedding CCCL JIT header bundle into ${EMBED_CCCL_HEADERS_OUT}"
  VERBATIM)

add_library(
  simpatico STATIC
  src/decode/decode_session.cpp
  # Generated: complete CCCL header bundle embedded for the runtime JIT.
  "${EMBED_CCCL_HEADERS_OUT}"
  # CUDA kernels: operators + offsets
  src/bridge/offsets_cumsum.cu
  src/selection/selection_wave.cu
  src/selection/chunk_row_set_build.cu
  src/selection/row_id_space.cu
  src/selection/chunk_row_set_convert.cu
  src/operators/alp_compressor.cu
  src/operators/alp_rd_compressor.cu
  src/operators/ans_compressor.cu
  src/operators/bitcomp_compressor.cu
  src/operators/cascaded_compressor.cu
  src/operators/snappy_compressor.cu
  src/operators/lz4_compressor.cu
  src/operators/deflate_compressor.cu
  src/operators/bitjoin_bitextract.cu
  src/operators/dictionary_compressor.cu
  src/operators/str_split_compressor.cu
  src/operators/nvcomp_batched_codec.cu
  src/operators/nvcomp_chunk_select.cu
  # Plain-CUDA JIT host
  src/jit/nvrtc_compiler.cpp
  src/jit/disk_cache.cpp
  src/jit/kernel_cache.cpp
  src/encode/jit/renderer.cpp
  src/decode/jit/renderer.cpp
  # Runtime: plan executor + bridge + API + rep glue
  src/selection/decompression_pushdown_policy.cpp
  src/plan/bit_spec.cpp
  src/plan/plan_dsl.cpp
  src/plan/plan_tree.cpp
  src/plan/operator_registry.cpp
  src/plan/representation_factory.cpp
  src/plan/bitjoin_layout.cpp
  src/plan/validity.cpp
  src/plan/compress.cpp
  src/plan/decompress.cpp
  src/util/stream_pool.cpp
  src/util/cudf_like.cu
  src/util/host_observation.cpp
  src/util/isoc23_compat.c
  src/bridge/fused_tree_build.cpp
  src/bridge/codegen_runtime.cpp
  src/simpatico_codegen.cpp
  src/api/compressed_table_io.cpp
  src/explore/compression_explorer.cpp)
# Keep libcudf's CCCL includes ahead of the toolkit's implicit headers. Imported
# -isystem paths would otherwise be searched after ordinary -I paths.
set_target_properties(
  simpatico
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             NO_SYSTEM_FROM_IMPORTED ON
             CXX_SCAN_FOR_MODULES OFF
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON)
if(VCPKG_BUILD)
  set_target_properties(simpatico PROPERTIES CUDA_RUNTIME_LIBRARY Static)
endif()
target_compile_options(
  simpatico
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda>
          $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>)
target_compile_features(simpatico PUBLIC cxx_std_20)
# Name of the NVTX domain simpatico's (and Sirius's) ranges are published into.
# The parent project sets this from its own project name; the fallback only
# applies to the standalone simpatico_codegen build, whose PROJECT_NAME is not
# "sirius". PUBLIC so consumers — sirius_extension and, transitively,
# sirius_unittest and parquet_benchmark — compile their copy of the tag type
# against the same name.
if(NOT DEFINED SIRIUS_NVTX_DOMAIN_NAME)
  set(SIRIUS_NVTX_DOMAIN_NAME "sirius")
endif()
target_compile_definitions(
  simpatico
  PUBLIC LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
         CCCL_DISABLE_WARPSPEED_SCAN CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER
         SIRIUS_NVTX_DOMAIN_NAME="${SIRIUS_NVTX_DOMAIN_NAME}")
# nvrtc_compiler.cpp #includes the generated embedded_headers.h from here.
add_dependencies(simpatico simpatico_embedded_headers)
target_include_directories(
  simpatico PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                    "${CMAKE_CURRENT_BINARY_DIR}/generated")
# In the Sirius vcpkg build the overlay nvrtc port provides a static NVRTC
# (nvrtc::nvrtc_static) so the extension carries no libnvrtc.so runtime dep.
# Other builds use the toolkit's shared NVRTC target.
if(VCPKG_BUILD AND NOT TARGET nvrtc::nvrtc_static)
  # A standalone vcpkg configure has no parent to discover this target.
  find_package(nvrtc REQUIRED CONFIG)
endif()
if(TARGET nvrtc::nvrtc_static)
  set(SIMPATICO_NVRTC_LIB nvrtc::nvrtc_static)
else()
  set(SIMPATICO_NVRTC_LIB CUDA::nvrtc)
endif()

if(VCPKG_BUILD)
  set(SIMPATICO_CUDART_LIB CUDA::cudart_static)
else()
  set(SIMPATICO_CUDART_LIB CUDA::cudart)
endif()
target_link_libraries(
  simpatico
  PUBLIC cudf::cudf rmm::rmm nvcomp::nvcomp ${SIMPATICO_CUDART_LIB}
         ${SIMPATICO_NVRTC_LIB} CUDA::cuda_driver
  PRIVATE simpatico_cache_identity)
if(SIMPATICO_ONPAIR)
  target_sources(simpatico PRIVATE src/operators/onpair_compressor.cu
                                   src/operators/onpair_like.cu)
  target_link_libraries(simpatico PRIVATE onpair_bridge)
  target_compile_definitions(simpatico PUBLIC SIMPATICO_ONPAIR=1)
endif()

# Sirius already provides cuCascade. Standalone Simpatico keeps its CUDA-only
# backend and does not acquire a dependency on the rest of the engine.
if(TARGET cuCascade::cucascade_static)
  target_link_libraries(simpatico PRIVATE cuCascade::cucascade_static)
  target_compile_definitions(simpatico
                             PRIVATE SIMPATICO_USE_CUCASCADE_STREAM_POOL)
endif()

function(add_codegen_host_test name)
  add_executable(${name} tests/${name}.cpp)
  set_target_properties(${name} PROPERTIES CXX_SCAN_FOR_MODULES OFF)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(${name} PRIVATE simpatico)
endfunction()

add_codegen_host_test(test_compress_with_plan_roundtrip)
if(SIMPATICO_ONPAIR)
  add_executable(test_onpair_k_gate tests/test_onpair_k_gate.cpp)
  target_include_directories(test_onpair_k_gate PRIVATE src)
  target_compile_features(test_onpair_k_gate PRIVATE cxx_std_17)
  add_codegen_host_test(test_onpair_roundtrip)
  add_codegen_host_test(test_onpair_like)
  add_codegen_host_test(test_decode_scratch)
  target_link_options(
    test_onpair_like PRIVATE "-Wl,--wrap=cudaEventCreate"
    "-Wl,--wrap=cudaStreamSynchronize" "-Wl,--wrap=cudaMemcpyAsync")
  target_link_options(
    test_onpair_roundtrip PRIVATE "-Wl,--wrap=cudaStreamQuery"
    "-Wl,--wrap=cudaStreamSynchronize" "-Wl,--wrap=cudaMemcpyAsync")
else()
  add_codegen_host_test(test_onpair_disabled)
endif()
add_codegen_host_test(test_async_decode)
target_link_options(test_async_decode PRIVATE "-Wl,--wrap=cudaStreamQuery"
                    "-Wl,--wrap=cudaStreamSynchronize")
add_codegen_host_test(test_scan_filter_session)
add_codegen_host_test(test_compressed_table_io)
add_codegen_host_test(test_leaf_describe)
add_codegen_host_test(test_plan_tree)
add_codegen_host_test(test_fused_tree_build)
add_codegen_host_test(test_jit_kernel_cache_plain)
add_codegen_host_test(test_jit_kernel_cache)
add_codegen_host_test(test_fused_operator_sweep)
add_codegen_host_test(test_operator_sweep)
if(SIMPATICO_ONPAIR)
  target_compile_definitions(test_operator_sweep PRIVATE SIMPATICO_ONPAIR=1)
endif()
add_codegen_host_test(test_representation_contract)
add_codegen_host_test(test_string_decode_move)
add_codegen_host_test(test_snappy_backend)
# The view contract crosses the engine converter boundary. Standalone Simpatico
# has no pin owner.
if(SIRIUS_BUILD_TESTS)
  add_codegen_host_test(test_hw_decompression_gate)
  target_include_directories(test_hw_decompression_gate
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  target_link_libraries(test_hw_decompression_gate PRIVATE sirius_core)
  target_link_options(
    test_hw_decompression_gate PRIVATE "LINKER:--allow-multiple-definition"
    "LINKER:--wrap=cudaDriverGetVersion")
  add_codegen_host_test(test_string_decode_view)
  target_include_directories(
    test_string_decode_view
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../test/cpp)
  target_link_libraries(test_string_decode_view PRIVATE sirius_core)
  # Both Rust bridges (telemetry whole-archive, onpair) carry compiler_builtins;
  # the engine's own executables resolve the duplicates the same way.
  # Count requests for one watched PlanTree independently of the scan's own diagnostics.
  target_link_options(test_string_decode_view PRIVATE
                      "LINKER:--allow-multiple-definition"
                      "LINKER:--wrap=_ZN9simpatico14decode_requestERKNS_21column_decode_requestERNS_12decode_frameE")
endif()
add_codegen_host_test(test_bitpack_layout_contract)
add_codegen_host_test(test_multi_gpu_stream_affinity)
if(TARGET cuCascade::cucascade_static)
  target_compile_definitions(test_multi_gpu_stream_affinity
                             PRIVATE SIMPATICO_USE_CUCASCADE_STREAM_POOL)
endif()
add_codegen_host_test(test_masked_decode_variants)
add_codegen_host_test(test_chunk_selective)
add_codegen_host_test(test_selection_wave)
add_codegen_host_test(test_chunk_row_set_build)
add_codegen_host_test(test_row_id_space)
add_codegen_host_test(test_chunk_row_set_convert)
add_codegen_host_test(test_decompress_column_rows)
add_codegen_host_test(test_render_signature_contract)

# Multi-mode CLI. Target is simpatico_cli to avoid clashing with the lib target;
# the installed binary is named 'simpatico'.
add_executable(simpatico_cli cli/simpatico_main.cpp)
set_target_properties(simpatico_cli PROPERTIES OUTPUT_NAME simpatico
                                               CXX_SCAN_FOR_MODULES OFF)
target_include_directories(simpatico_cli
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(simpatico_cli PRIVATE simpatico)

enable_testing()
add_subdirectory(tests/cache)
add_test(NAME compress_with_plan_roundtrip
         COMMAND test_compress_with_plan_roundtrip)
if(SIMPATICO_ONPAIR)
  foreach(lanes 1 2 4)
    add_test(
      NAME decode_scratch_${lanes}_lanes
      COMMAND
        "${CMAKE_COMMAND}" "-DDECODE_EXE=$<TARGET_FILE:test_decode_scratch>"
        "-DRESULT_DIR=${CMAKE_CURRENT_BINARY_DIR}/decode-scratch-${lanes}"
        "-DLANES=${lanes}" -P
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_decode_scratch.cmake")
    set_property(
      TEST decode_scratch_${lanes}_lanes
      PROPERTY ENVIRONMENT
               "SIRIUS_EXP_FUSED_SCAN_COST_RULE=0"
               "SIRIUS_EXP_FUSED_SCAN_STAGE_BYTES=67108864"
               "SIRIUS_EXP_ONPAIR_LIKE=1"
               "SIRIUS_EXP_FUSED_SCAN_FILTER=1"
               "SIRIUS_EXP_ONPAIR_SELECTIVE=1"
               "SIRIUS_EXP_FUSED_SCAN_MAX_SEL=0.35"
               "SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL=0.10"
               "SIRIUS_EXP_ONPAIR_LIKE_MASK_ABOVE_CAP=1"
               "SIRIUS_EXP_FUSED_SCAN_BATCHED_WAVE2=1")
  endforeach()
  add_test(NAME onpair_k6_gate_unset
           COMMAND ${CMAKE_COMMAND} -E env --unset=SIRIUS_EXP_ONPAIR_K
                   $<TARGET_FILE:test_onpair_k_gate> 4)
  foreach(k IN ITEMS 4 6)
    add_test(NAME onpair_k6_gate_${k} COMMAND test_onpair_k_gate ${k})
    set_tests_properties(onpair_k6_gate_${k} PROPERTIES ENVIRONMENT "SIRIUS_EXP_ONPAIR_K=${k}")
  endforeach()
  foreach(value IN ITEMS invalid empty)
    if(value STREQUAL "empty")
      set(k "")
    else()
      set(k "5")
    endif()
    add_test(NAME onpair_k6_gate_${value} COMMAND test_onpair_k_gate invalid)
    set_tests_properties(onpair_k6_gate_${value} PROPERTIES ENVIRONMENT "SIRIUS_EXP_ONPAIR_K=${k}")
  endforeach()
  add_test(NAME onpair_k6_roundtrip COMMAND test_onpair_roundtrip --k6)
  add_test(NAME onpair_k6_code_65535_high_queue COMMAND test_onpair_roundtrip --k6-code-65535)
  set_tests_properties(onpair_k6_roundtrip onpair_k6_code_65535_high_queue PROPERTIES ENVIRONMENT "SIRIUS_EXP_ONPAIR_K=6")
  add_test(NAME onpair_k6_cross_gate
           COMMAND ${CMAKE_COMMAND} -DONPAIR_EXE=$<TARGET_FILE:test_onpair_roundtrip>
                   -DONPAIR_FILE=${CMAKE_CURRENT_BINARY_DIR}/onpair_k6_cross_gate.hpln
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_onpair_cross_gate.cmake)
  add_test(NAME onpair_k6_like COMMAND test_onpair_like --k6)
  add_test(NAME onpair_k6_like_batch_boundary COMMAND test_onpair_like --k6-batch-boundary)
  set_tests_properties(onpair_k6_like onpair_k6_like_batch_boundary PROPERTIES ENVIRONMENT
    "SIRIUS_EXP_ONPAIR_K=6;SIRIUS_EXP_ONPAIR_LIKE=1;SIRIUS_EXP_FUSED_SCAN_FILTER=1;SIRIUS_EXP_ONPAIR_SELECTIVE=1")
  add_test(NAME onpair_k6_selective COMMAND test_masked_decode_variants --onpair-k6)
  add_test(NAME onpair_k6_selective_final_token COMMAND test_masked_decode_variants --onpair-k6-final-token)
  set_tests_properties(onpair_k6_selective onpair_k6_selective_final_token PROPERTIES
    SKIP_REGULAR_EXPRESSION "SKIP: no CUDA device"
    ENVIRONMENT
    "SIRIUS_EXP_ONPAIR_K=6;SIRIUS_EXP_ONPAIR_SELECTIVE=1")
  add_test(NAME onpair_roundtrip COMMAND test_onpair_roundtrip)
  add_test(NAME onpair_like COMMAND test_onpair_like)
  set_tests_properties(
    onpair_like
    PROPERTIES
      ENVIRONMENT
      "SIRIUS_EXP_ONPAIR_LIKE=1;SIRIUS_EXP_FUSED_SCAN_FILTER=1;SIRIUS_EXP_ONPAIR_SELECTIVE=1"
  )
  add_test(NAME onpair_like_switch_off COMMAND test_onpair_like --off)
  foreach(batched 0 1)
    add_test(NAME onpair_like_wave2_${batched} COMMAND test_onpair_like --wave2)
    set_tests_properties(onpair_like_wave2_${batched} PROPERTIES ENVIRONMENT
      "SIRIUS_EXP_ONPAIR_LIKE=1;SIRIUS_EXP_FUSED_SCAN_FILTER=1;SIRIUS_EXP_ONPAIR_SELECTIVE=1;SIRIUS_EXP_FUSED_SCAN_BATCHED_WAVE2=${batched};SIRIUS_EXP_FUSED_SCAN_ZERO_SKIP=1")
    add_test(NAME onpair_like_wave2_${batched}_zero_skip_off COMMAND test_onpair_like --wave2)
    set_tests_properties(onpair_like_wave2_${batched}_zero_skip_off PROPERTIES ENVIRONMENT
      "SIRIUS_EXP_ONPAIR_LIKE=1;SIRIUS_EXP_FUSED_SCAN_FILTER=1;SIRIUS_EXP_ONPAIR_SELECTIVE=1;SIRIUS_EXP_FUSED_SCAN_BATCHED_WAVE2=${batched};SIRIUS_EXP_FUSED_SCAN_ZERO_SKIP=0")
  endforeach()
  foreach(batched 0 1)
    foreach(skip 0 1)
      # The BOOL8 above-cap switch follows zero skip: each value covers its delivery and decline.
      add_test(NAME onpair_like_deferred_batched_${batched}_zero_skip_${skip}
               COMMAND test_onpair_like --deferred)
      set_tests_properties(onpair_like_deferred_batched_${batched}_zero_skip_${skip}
        PROPERTIES ENVIRONMENT
        "SIRIUS_EXP_ONPAIR_LIKE=1;SIRIUS_EXP_FUSED_SCAN_FILTER=1;SIRIUS_EXP_ONPAIR_SELECTIVE=1;SIRIUS_EXP_FUSED_SCAN_BATCHED_WAVE2=${batched};SIRIUS_EXP_FUSED_SCAN_ZERO_SKIP=${skip};SIRIUS_EXP_FUSED_SCAN_MAX_SEL=0.35;SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL=0.10;SIRIUS_EXP_ONPAIR_LIKE_MASK_ABOVE_CAP=${skip}")
    endforeach()
  endforeach()
  foreach(cost 0 1)
    add_test(NAME onpair_like_cost_rule_${cost} COMMAND test_onpair_like --cost)
    set_tests_properties(onpair_like_cost_rule_${cost} PROPERTIES ENVIRONMENT
      "SIRIUS_EXP_ONPAIR_LIKE=1;SIRIUS_EXP_FUSED_SCAN_FILTER=1;SIRIUS_EXP_ONPAIR_SELECTIVE=1;SIRIUS_EXP_FUSED_SCAN_COST_RULE=${cost}")
  endforeach()
  foreach(mask 0 1)
    add_test(NAME onpair_like_mask_above_cap_${mask} COMMAND test_onpair_like --mask)
    set_tests_properties(onpair_like_mask_above_cap_${mask} PROPERTIES ENVIRONMENT
      "SIRIUS_EXP_ONPAIR_LIKE=1;SIRIUS_EXP_FUSED_SCAN_FILTER=1;SIRIUS_EXP_ONPAIR_SELECTIVE=1;SIRIUS_EXP_ONPAIR_LIKE_MASK_ABOVE_CAP=${mask}")
  endforeach()
  set_tests_properties(
    onpair_like_switch_off
    PROPERTIES ENVIRONMENT
               "SIRIUS_EXP_ONPAIR_LIKE=0;SIRIUS_EXP_FUSED_SCAN_FILTER=1")
  if(SIMPATICO_LARGE_MEMORY_TESTS)
    add_test(NAME onpair_promoted_offsets COMMAND test_onpair_roundtrip
                                                  --promoted-offsets)
    set_tests_properties(
      onpair_promoted_offsets
      PROPERTIES LABELS large_memory RUN_SERIAL TRUE ENVIRONMENT
                 "SIRIUS_EXP_ONPAIR_SELECTIVE=1")
  endif()
else()
  add_test(NAME onpair_disabled COMMAND test_onpair_disabled)
endif()
add_test(NAME async_decode COMMAND test_async_decode)
foreach(batched 0 1)
  foreach(skip 0 1)
    add_test(NAME scan_filter_session_batched_${batched}_zero_skip_${skip}
             COMMAND test_scan_filter_session)
    set_tests_properties(scan_filter_session_batched_${batched}_zero_skip_${skip}
      PROPERTIES ENVIRONMENT
      "SIRIUS_EXP_FUSED_SCAN_BATCHED_WAVE2=${batched};SIRIUS_EXP_FUSED_SCAN_ZERO_SKIP=${skip}")
  endforeach()
endforeach()
add_test(NAME compressed_table_io COMMAND test_compressed_table_io)
add_test(NAME leaf_describe COMMAND test_leaf_describe)
add_test(NAME plan_tree COMMAND test_plan_tree)
add_test(NAME fused_tree_build COMMAND test_fused_tree_build)
add_test(NAME jit_kernel_cache_plain COMMAND test_jit_kernel_cache_plain)
add_test(NAME jit_kernel_cache COMMAND test_jit_kernel_cache)
add_test(NAME fused_operator_sweep COMMAND test_fused_operator_sweep)
add_test(NAME operator_sweep COMMAND test_operator_sweep)
add_test(NAME representation_contract COMMAND test_representation_contract)
foreach(backend default cuda hardware)
  add_test(NAME snappy_backend_${backend}
           COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_SNAPPY_BACKEND=${backend}
                   $<TARGET_FILE:test_snappy_backend> ${backend})
endforeach()
add_test(NAME snappy_backend_unset
         COMMAND ${CMAKE_COMMAND} -E env --unset=SIRIUS_EXP_SNAPPY_BACKEND
                 $<TARGET_FILE:test_snappy_backend> default)
foreach(value invalid "" CUDA)
  # Give the empty value a stable, nonempty CTest name.
  set(suffix "${value}")
  if(value STREQUAL "")
    set(suffix empty)
  endif()
  add_test(NAME snappy_backend_${suffix}
           COMMAND ${CMAKE_COMMAND} -E env "SIRIUS_EXP_SNAPPY_BACKEND=${value}"
                   $<TARGET_FILE:test_snappy_backend> invalid)
endforeach()
set_tests_properties(snappy_backend_hardware PROPERTIES SKIP_RETURN_CODE 77)
if(SIRIUS_BUILD_TESTS)
  add_test(NAME hw_decompression_gate COMMAND test_hw_decompression_gate)
endif()
add_test(NAME string_decode_move
         COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_STRING_DECODE_COPY=0
                 $<TARGET_FILE:test_string_decode_move>)
add_test(NAME string_decode_move_copy
         COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_STRING_DECODE_COPY=1
                 $<TARGET_FILE:test_string_decode_move> --copy)
if(SIRIUS_BUILD_TESTS)
  foreach(mode unset 0 1 capped no_frames)
    set(gate SIRIUS_EXP_DECODED_LIKE=1)
    set(frames 1)
    set(cap 1)
    if(mode STREQUAL "unset")
      set(gate --unset=SIRIUS_EXP_DECODED_LIKE)
    elseif(mode STREQUAL "0")
      set(gate SIRIUS_EXP_DECODED_LIKE=0)
    elseif(mode STREQUAL "capped")
      set(cap 0.10)
    elseif(mode STREQUAL "no_frames")
      set(frames 0)
    endif()
    add_test(NAME decoded_like_${mode}
             COMMAND ${CMAKE_COMMAND} -E env ${gate} SIRIUS_EXP_ONPAIR_LIKE=0
                     SIRIUS_EXP_CHUNK_SELECTIVE=${frames} SIRIUS_EXP_FUSED_SCAN_FILTER=1
                     SIRIUS_EXP_FUSED_SCAN_MAX_SEL=${cap} SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL=${cap}
                     SIRIUS_EXP_FUSED_SCAN_COST_RULE=0 SIRIUS_EXP_STRING_DECODE_COPY=0
                     $<TARGET_FILE:test_string_decode_view> --decoded-like ${mode})
  endforeach()
  foreach(mode unset 0 1 arena_off)
    set(gate SIRIUS_EXP_CHUNK_SELECTIVE=${mode})
    set(arena SIRIUS_EXP_DECODE_SCRATCH_ARENA=1)
    if(mode STREQUAL "unset")
      set(gate --unset=SIRIUS_EXP_CHUNK_SELECTIVE)
    elseif(mode STREQUAL "arena_off")
      set(gate SIRIUS_EXP_CHUNK_SELECTIVE=1)
      set(arena SIRIUS_EXP_DECODE_SCRATCH_ARENA=0)
    endif()
    add_test(NAME chunk_selective_admission_${mode}
             COMMAND ${CMAKE_COMMAND} -E env ${gate} ${arena} SIRIUS_EXP_STRING_DECODE_COPY=0
                     SIRIUS_EXP_FUSED_SCAN_FILTER=1 SIRIUS_EXP_FUSED_SCAN_COST_RULE=0
                     SIRIUS_EXP_FUSED_SCAN_LATCH_ZERO_CHECK=0 SIRIUS_EXP_FUSED_SCAN_MAX_SEL=1
                     SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL=0.10 $<TARGET_FILE:test_string_decode_view> --admission)
    if(NOT mode STREQUAL "arena_off")
      add_test(NAME chunk_selective_remote_${mode}
               COMMAND ${CMAKE_COMMAND} -E env ${gate} ${arena} SIRIUS_EXP_STRING_DECODE_COPY=0
                       SIRIUS_EXP_FUSED_SCAN_FILTER=1 $<TARGET_FILE:test_string_decode_view> --cross-gpu-admission)
      set_tests_properties(chunk_selective_remote_${mode} PROPERTIES SKIP_RETURN_CODE 77)
    endif()
  endforeach()
  add_test(NAME chunk_selective_arena_bytes
           COMMAND ${CMAKE_COMMAND} -DPROGRAM=$<TARGET_FILE:test_string_decode_view>
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_chunk_selective_arena.cmake)
  foreach(codec snappy ans lz4 deflate)
    set(suffix "_${codec}")
    if(codec STREQUAL "snappy")
      set(suffix "")
    endif()
    foreach(enabled 0 1)
      add_test(NAME chunk_selective_pinned_${enabled}${suffix}
               COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_CHUNK_SELECTIVE=${enabled}
                       SIRIUS_EXP_STRING_DECODE_COPY=0 SIRIUS_EXP_FUSED_SCAN_FILTER=1
                       SIRIUS_EXP_FUSED_SCAN_MAX_SEL=1 $<TARGET_FILE:test_string_decode_view> --framed ${codec})
    endforeach()
    if(SIMPATICO_LARGE_MEMORY_TESTS)
      add_test(NAME chunk_selective_large_fallback${suffix}
               COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_CHUNK_SELECTIVE=1 SIRIUS_EXP_STRING_DECODE_COPY=0
                       $<TARGET_FILE:test_string_decode_view> --large-framed ${codec})
      set_tests_properties(chunk_selective_large_fallback${suffix} PROPERTIES LABELS large_memory RUN_SERIAL TRUE)
    endif()
  endforeach()
  add_test(
    NAME string_decode_view
    COMMAND
      ${CMAKE_COMMAND} -E env SIRIUS_EXP_STRING_DECODE_COPY=0
      SIRIUS_EXP_FUSED_SCAN_FILTER=0 $<TARGET_FILE:test_string_decode_view>)
  add_test(
    NAME string_decode_view_copy
    COMMAND
      ${CMAKE_COMMAND} -E env SIRIUS_EXP_STRING_DECODE_COPY=1
      SIRIUS_EXP_FUSED_SCAN_FILTER=0 $<TARGET_FILE:test_string_decode_view>
      --copy)
  add_test(
    NAME string_decode_view_late_mat
    COMMAND
      ${CMAKE_COMMAND} -E env SIRIUS_EXP_STRING_DECODE_COPY=0
      SIRIUS_EXP_FUSED_SCAN_FILTER=1 SIRIUS_EXP_LATE_MAT=1
      $<TARGET_FILE:test_string_decode_view>)
  add_test(
    NAME string_decode_view_late_mat_copy
    COMMAND
      ${CMAKE_COMMAND} -E env SIRIUS_EXP_STRING_DECODE_COPY=1
      SIRIUS_EXP_FUSED_SCAN_FILTER=1 SIRIUS_EXP_LATE_MAT=1
      $<TARGET_FILE:test_string_decode_view> --copy)
  if(SIMPATICO_LARGE_MEMORY_TESTS)
    add_test(NAME string_decode_view_large
             COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_STRING_DECODE_COPY=0
                     $<TARGET_FILE:test_string_decode_view> --large)
    add_test(NAME string_decode_view_large_copy
             COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_STRING_DECODE_COPY=1
                     $<TARGET_FILE:test_string_decode_view> --large --copy)
    set_tests_properties(string_decode_view_large string_decode_view_large_copy
                         PROPERTIES LABELS large_memory RUN_SERIAL TRUE)
  endif()
endif()
add_test(NAME bitpack_layout_contract COMMAND test_bitpack_layout_contract)
add_test(NAME multi_gpu_stream_affinity COMMAND test_multi_gpu_stream_affinity)
add_test(NAME masked_decode_variants COMMAND test_masked_decode_variants)
if(SIMPATICO_ONPAIR)
  # Separate processes exercise the read-once gate, including rejection before
  # any CUDA work.
  set_tests_properties(masked_decode_variants
                       PROPERTIES ENVIRONMENT "SIRIUS_EXP_ONPAIR_SELECTIVE=1")
  add_test(NAME onpair_selective_default
           COMMAND ${CMAKE_COMMAND} -E env --unset=SIRIUS_EXP_ONPAIR_SELECTIVE
                   $<TARGET_FILE:test_masked_decode_variants> --onpair-only)
  add_test(
    NAME onpair_selective_off
    COMMAND
      ${CMAKE_COMMAND}
      -DONPAIR_MASKED_EXE=$<TARGET_FILE:test_masked_decode_variants>
      -DONPAIR_RESULT_DIR=${CMAKE_CURRENT_BINARY_DIR}/onpair-selective-off -P
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_onpair_selective_off.cmake)
  foreach(value invalid "" ON)
    set(suffix "${value}")
    if(value STREQUAL "")
      set(suffix empty)
    endif()
    add_test(
      NAME onpair_selective_${suffix}
      COMMAND ${CMAKE_COMMAND} -E env "SIRIUS_EXP_ONPAIR_SELECTIVE=${value}"
              $<TARGET_FILE:test_masked_decode_variants> --onpair-invalid)
  endforeach()
endif()
# Fresh processes exercise the read-once SIRIUS_EXP_CHUNK_SELECTIVE switch on both CUDA-capable
# Snappy backends, the unset default, the switch-off bytes, and rejected values.
foreach(backend default cuda)
  add_test(NAME chunk_selective_snappy_${backend}
           COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_CHUNK_SELECTIVE=1
                   SIRIUS_EXP_SNAPPY_BACKEND=${backend}
                   $<TARGET_FILE:test_chunk_selective>)
endforeach()
add_test(NAME chunk_selective_offsets_width
         COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_CHUNK_SELECTIVE=1
                 LIBCUDF_LARGE_STRINGS_ENABLED=1 LIBCUDF_LARGE_STRINGS_THRESHOLD=65536
                 $<TARGET_FILE:test_chunk_selective> --wide)
add_test(NAME chunk_selective_snappy_hardware
         COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_CHUNK_SELECTIVE=1
                 SIRIUS_EXP_SNAPPY_BACKEND=hardware $<TARGET_FILE:test_chunk_selective> --hardware)
set(chunk_probe_file "${CMAKE_CURRENT_BINARY_DIR}/chunk-selective-probe.hpln")
add_test(NAME chunk_selective_write_probe
         COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_CHUNK_SELECTIVE=0
                 SIRIUS_EXP_SNAPPY_BACKEND=cuda $<TARGET_FILE:test_chunk_selective> --write-probe "${chunk_probe_file}")
set_tests_properties(chunk_selective_write_probe PROPERTIES FIXTURES_SETUP chunk_probe)
add_test(NAME chunk_selective_clean_probe COMMAND ${CMAKE_COMMAND} -E rm -f "${chunk_probe_file}")
set_tests_properties(chunk_selective_clean_probe PROPERTIES FIXTURES_CLEANUP chunk_probe)
add_test(NAME chunk_selective_off_probe
         COMMAND ${CMAKE_COMMAND} -E env SIRIUS_EXP_CHUNK_SELECTIVE=0
                 SIRIUS_EXP_SNAPPY_BACKEND=invalid $<TARGET_FILE:test_chunk_selective> --probe-off "${chunk_probe_file}")
set_tests_properties(chunk_selective_off_probe PROPERTIES FIXTURES_REQUIRED chunk_probe)
add_test(NAME chunk_selective_unset
         COMMAND ${CMAKE_COMMAND} -E env --unset=SIRIUS_EXP_CHUNK_SELECTIVE
                 $<TARGET_FILE:test_chunk_selective>)
add_test(
  NAME chunk_selective_off
  COMMAND
    ${CMAKE_COMMAND} -DCHUNK_SELECTIVE_EXE=$<TARGET_FILE:test_chunk_selective>
    -DCHUNK_SELECTIVE_RESULT_DIR=${CMAKE_CURRENT_BINARY_DIR}/chunk-selective-off
    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_chunk_selective_off.cmake)
set_tests_properties(chunk_selective_off PROPERTIES SKIP_REGULAR_EXPRESSION
                     "chunk-selective comparison skipped [(]no CUDA device[)]")
set_tests_properties(chunk_selective_snappy_default chunk_selective_snappy_cuda
                     chunk_selective_snappy_hardware chunk_selective_offsets_width
                     chunk_selective_off_probe chunk_selective_write_probe chunk_selective_unset
                     PROPERTIES SKIP_RETURN_CODE 77)
foreach(value invalid "" ON)
  set(suffix "${value}")
  if(value STREQUAL "")
    set(suffix empty)
  endif()
  add_test(NAME chunk_selective_${suffix}
           COMMAND ${CMAKE_COMMAND} -E env "SIRIUS_EXP_CHUNK_SELECTIVE=${value}"
                   $<TARGET_FILE:test_chunk_selective> --invalid)
endforeach()
add_test(NAME selection_wave COMMAND test_selection_wave)
add_test(NAME chunk_row_set_build COMMAND test_chunk_row_set_build)
add_test(NAME row_id_space COMMAND test_row_id_space)
add_test(NAME chunk_row_set_convert COMMAND test_chunk_row_set_convert)
add_test(NAME decompress_column_rows COMMAND test_decompress_column_rows)
add_test(NAME render_signature_contract COMMAND test_render_signature_contract)
