# SPDX-License-Identifier: Apache-2.0
# Compile-only compatibility check. Run on a configured CUDA build host:
# cmake -DSIMPATICO_OFF_BINARY_DIR=/tmp/simpatico-off \
#   '-DSIMPATICO_OFF_CONFIGURE_ARGS=-DCMAKE_PREFIX_PATH=...;-DCMAKE_CUDA_ARCHITECTURES=120' \
#   -P tests/check_onpair_off_build.cmake
# No GPU tests are executed. In the resulting build, CTest onpair_disabled checks
# the clear rejection of OnPair plans and v16 file headers.
cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED SIMPATICO_OFF_BINARY_DIR)
  message(FATAL_ERROR "Set SIMPATICO_OFF_BINARY_DIR to a separate OFF build directory")
endif()
get_filename_component(simpatico_source_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${simpatico_source_dir}"
          -B "${SIMPATICO_OFF_BINARY_DIR}" ${SIMPATICO_OFF_CONFIGURE_ARGS}
          -DSIMPATICO_ONPAIR=OFF
  RESULT_VARIABLE configure_status)
if(NOT configure_status EQUAL 0)
  message(FATAL_ERROR "OnPair OFF configure failed: ${configure_status}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${SIMPATICO_OFF_BINARY_DIR}"
          --target simpatico test_onpair_disabled test_masked_decode_variants
  RESULT_VARIABLE build_status)
if(NOT build_status EQUAL 0)
  message(FATAL_ERROR "OnPair OFF compile check failed: ${build_status}")
endif()
