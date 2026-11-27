# SPDX-License-Identifier: Apache-2.0

# Compare chunk-selective bytes with switch-off full-decode-plus-gather bytes in
# fresh processes.
cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CHUNK_SELECTIVE_EXE OR NOT DEFINED CHUNK_SELECTIVE_RESULT_DIR)
  message(FATAL_ERROR "Set CHUNK_SELECTIVE_EXE and CHUNK_SELECTIVE_RESULT_DIR")
endif()
file(MAKE_DIRECTORY "${CHUNK_SELECTIVE_RESULT_DIR}")
foreach(enabled 1 0)
  set(CHUNK_RESULT_FILE "${CHUNK_SELECTIVE_RESULT_DIR}/selected-${enabled}.bin")
  file(REMOVE "${CHUNK_RESULT_FILE}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "SIRIUS_EXP_CHUNK_SELECTIVE=${enabled}"
            "${CHUNK_SELECTIVE_EXE}" --record "${CHUNK_RESULT_FILE}"
    RESULT_VARIABLE CHUNK_STATUS
    OUTPUT_VARIABLE CHUNK_OUTPUT
    ERROR_VARIABLE CHUNK_ERRORS)
  message("${CHUNK_OUTPUT}${CHUNK_ERRORS}")
  if(CHUNK_STATUS STREQUAL "77")
    file(REMOVE_RECURSE "${CHUNK_SELECTIVE_RESULT_DIR}")
    message("chunk-selective comparison skipped (no CUDA device)")
    return()
  endif()
  if(NOT CHUNK_STATUS STREQUAL "0")
    message(FATAL_ERROR "chunk-selective switch=${enabled} failed: ${CHUNK_STATUS}")
  endif()
  if(NOT EXISTS "${CHUNK_RESULT_FILE}")
    message(
      FATAL_ERROR "chunk-selective switch=${enabled} did not produce selected bytes")
  endif()
  file(SIZE "${CHUNK_RESULT_FILE}" CHUNK_SIZE)
  if(CHUNK_SIZE EQUAL 0)
    message(FATAL_ERROR "chunk-selective switch=${enabled} produced an empty record")
  endif()
endforeach()
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E compare_files
    "${CHUNK_SELECTIVE_RESULT_DIR}/selected-1.bin"
    "${CHUNK_SELECTIVE_RESULT_DIR}/selected-0.bin"
  RESULT_VARIABLE CHUNK_COMPARISON)
if(NOT CHUNK_COMPARISON STREQUAL "0")
  message(
    FATAL_ERROR
      "chunk-selective switch-off full selection differs from switch-on selected bytes")
endif()
file(REMOVE_RECURSE "${CHUNK_SELECTIVE_RESULT_DIR}")
message("PASS: chunk-selective switch-on/off selected offsets and chars match byte-for-byte")
