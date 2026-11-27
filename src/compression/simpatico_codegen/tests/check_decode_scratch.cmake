# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
file(MAKE_DIRECTORY "${RESULT_DIR}")
# capacity_projection exercises 65,535 rows / four lanes, 131,071 / two, and 262,143 / one.
set(MODES projection capacity_projection like full above zero deferred
          shared_deferred)
if(LANES EQUAL 4)
  list(APPEND MODES url_projection url_like url_like_parallel)
endif()
foreach(mode IN LISTS MODES)
  # Also exercise selected projection on all four lanes, with the existing unbatched path.
  set(BATCHED 1)
  if(mode STREQUAL "url_like_parallel")
    set(BATCHED 0)
  endif()
  foreach(enabled 0 1)
    execute_process(
      COMMAND
        "${CMAKE_COMMAND}" -E env "SIRIUS_EXP_DECODE_SCRATCH_ARENA=${enabled}"
        "SIRIUS_EXP_FUSED_SCAN_BATCHED_WAVE2=${BATCHED}"
        "${DECODE_EXE}" "${mode}" "${RESULT_DIR}/${mode}-${enabled}.bin"
        "${LANES}"
      RESULT_VARIABLE status
      OUTPUT_VARIABLE output
      ERROR_VARIABLE errors)
    message("${mode} arena=${enabled}: ${output}${errors}")
    if(NOT status EQUAL 0 OR NOT output MATCHES
                             "allocations=([0-9]+) peak=([0-9]+) retirements=0 backing=0 frees=([0-9]+)")
      message(
        FATAL_ERROR "decode scratch ${mode} arena=${enabled} failed: ${status}")
    endif()
    set(COUNT_${enabled} "${CMAKE_MATCH_1}")
    set(PEAK_${enabled} "${CMAKE_MATCH_2}")
    if(NOT CMAKE_MATCH_1 EQUAL CMAKE_MATCH_3)
      message(FATAL_ERROR "${mode}: allocations were retained after cleanup")
    endif()
  endforeach()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${RESULT_DIR}/${mode}-0.bin"
            "${RESULT_DIR}/${mode}-1.bin" RESULT_VARIABLE status)
  if(NOT status EQUAL 0 OR NOT COUNT_1 LESS COUNT_0)
    message(
      FATAL_ERROR
        "${mode}: bytes differ or allocations stayed ${COUNT_0} -> ${COUNT_1}")
  endif()
  math(EXPR PEAK_BOUND "${PEAK_0} + 1048576")
  if(PEAK_1 GREATER PEAK_BOUND)
    message(FATAL_ERROR "${mode}: arena exceeded the old peak + 1 MiB")
  endif()
  # Whole-decode replay is forbidden. The accepted peak is the baseline plus
  # at most 1 MiB of retained arena capacity. Exact-request OOM fallback and
  # non-OOM propagation are checked separately by arena_ordering().
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env "SIRIUS_EXP_DECODE_SCRATCH_ARENA=1"
      "SIRIUS_EXP_FUSED_SCAN_BATCHED_WAVE2=${BATCHED}"
      "${DECODE_EXE}" "${mode}" "${RESULT_DIR}/${mode}-limited.bin" "${LANES}"
      "${PEAK_BOUND}"
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE errors)
  message("${mode} bounded-peak limit: ${output}${errors}")
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "${mode}: memory-limited decode failed")
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${RESULT_DIR}/${mode}-0.bin"
            "${RESULT_DIR}/${mode}-limited.bin" RESULT_VARIABLE status)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "${mode}: memory-limited result differs")
  endif()
endforeach()
file(REMOVE_RECURSE "${RESULT_DIR}")
