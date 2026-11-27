# SPDX-License-Identifier: Apache-2.0

# Compare selected-route bytes with switch-off full-decode-plus-gather bytes in
# fresh processes.
cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED ONPAIR_MASKED_EXE OR NOT DEFINED ONPAIR_RESULT_DIR)
  message(FATAL_ERROR "Set ONPAIR_MASKED_EXE and ONPAIR_RESULT_DIR")
endif()
file(MAKE_DIRECTORY "${ONPAIR_RESULT_DIR}")
foreach(enabled 1 0)
  set(ONPAIR_RESULT_FILE "${ONPAIR_RESULT_DIR}/selected-${enabled}.bin")
  file(REMOVE "${ONPAIR_RESULT_FILE}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "SIRIUS_EXP_ONPAIR_SELECTIVE=${enabled}"
            "${ONPAIR_MASKED_EXE}" --onpair-only "${ONPAIR_RESULT_FILE}"
    RESULT_VARIABLE ONPAIR_STATUS
    OUTPUT_VARIABLE ONPAIR_OUTPUT
    ERROR_VARIABLE ONPAIR_ERRORS)
  message("${ONPAIR_OUTPUT}${ONPAIR_ERRORS}")
  if(NOT ONPAIR_STATUS EQUAL 0)
    message(FATAL_ERROR "OnPair switch=${enabled} failed: ${ONPAIR_STATUS}")
  endif()
  if(ONPAIR_OUTPUT MATCHES "SKIP: no CUDA device")
    file(REMOVE_RECURSE "${ONPAIR_RESULT_DIR}")
    return()
  endif()
  if(NOT EXISTS "${ONPAIR_RESULT_FILE}")
    message(
      FATAL_ERROR "OnPair switch=${enabled} did not produce selected bytes")
  endif()
endforeach()
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E compare_files "${ONPAIR_RESULT_DIR}/selected-1.bin"
    "${ONPAIR_RESULT_DIR}/selected-0.bin"
  RESULT_VARIABLE ONPAIR_COMPARISON)
if(NOT ONPAIR_COMPARISON EQUAL 0)
  message(
    FATAL_ERROR
      "OnPair switch-off full selection differs from switch-on selected bytes")
endif()
file(REMOVE_RECURSE "${ONPAIR_RESULT_DIR}")
message(
  "PASS: OnPair switch-on/off selected offsets and chars match byte-for-byte")
