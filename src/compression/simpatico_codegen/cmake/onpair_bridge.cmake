# SPDX-License-Identifier: Apache-2.0
# Rust `onpair` encoder bridge (rust/onpair-bridge), built by cargo through Corrosion and
# exposed as the `onpair_bridge` target (header in rust/onpair-bridge/include). Simpatico's
# `onpair` operator trains and encodes on the host through it; GPU decode is the operator's
# own kernel. Sirius fetches Corrosion for its telemetry bridge first; FIND_PACKAGE_ARGS lets
# a standalone Simpatico build pick up an installed Corrosion or fetch the same version.
include(FetchContent)
FetchContent_Declare(
  Corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.6.1
  FIND_PACKAGE_ARGS 0.6 CONFIG)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(
  MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/rust/onpair-bridge/Cargo.toml"
  CRATES onpair-bridge
  CRATE_TYPES staticlib
  PROFILE release)

target_include_directories(
  onpair_bridge
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rust/onpair-bridge/include>)

# Corrosion's installer exports the archive location but not its native links.
get_target_property(onpair_bridge_native_libraries onpair_bridge-static
                    INTERFACE_LINK_LIBRARIES)
if(VCPKG_BUILD)
  list(REMOVE_ITEM onpair_bridge_native_libraries gcc_s)
endif()
if(onpair_bridge_native_libraries)
  target_link_libraries(onpair_bridge INTERFACE ${onpair_bridge_native_libraries})
endif()
