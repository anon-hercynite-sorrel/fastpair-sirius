/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Internal header: only include from .cpp files that have the simpatico include path.

#include <api/compressed_leaf_slab.hpp>
#include <api/simpatico_codegen.hpp>

namespace sirius {

// Destroy the reconstructed channels before the resource and storage they reference.
struct compressed_device_blob {
  simpatico::compressed_leaf_slab leaves;
  simpatico::compressed_table table;
};

}  // namespace sirius
