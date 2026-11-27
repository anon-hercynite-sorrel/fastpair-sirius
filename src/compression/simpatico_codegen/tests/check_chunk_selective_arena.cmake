# Each switch is sampled once per process.
foreach(mode 0 1)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env SIRIUS_EXP_CHUNK_SELECTIVE=1 SIRIUS_EXP_FUSED_SCAN_FILTER=1
            SIRIUS_EXP_STRING_DECODE_COPY=0 SIRIUS_EXP_DECODE_SCRATCH_ARENA=${mode}
            "${PROGRAM}" --admission-bytes
    RESULT_VARIABLE status OUTPUT_VARIABLE output)
  if(NOT status STREQUAL "0" OR NOT output MATCHES "ADMISSION_BYTES=([0-9]+)")
    message(FATAL_ERROR "Failed to measure arena=${mode} reservation (${status}): ${output}")
  endif()
  set(bytes_${mode} "${CMAKE_MATCH_1}")
endforeach()
math(EXPR difference "${bytes_1} - ${bytes_0}")
if(NOT difference EQUAL 1048576)
  message(FATAL_ERROR "Two-frame admission changed by ${difference} bytes with the arena enabled")
endif()
