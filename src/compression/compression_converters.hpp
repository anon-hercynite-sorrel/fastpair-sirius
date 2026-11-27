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

#include <cucascade/data/representation_converter.hpp>

#include <cstddef>
#include <memory>

namespace sirius {

class decompression_pushdown_scan;

/**
 * @brief Per projected column of a per-batch host read, whether its OnPair preparation may wait
 * for the decode that needs it.
 *
 * A column the scan evaluates in wave 1 (LIKE, or a join filter's membership probes) is prepared
 * at read time: each wave-1 request and the value decode would otherwise prepare it again. Null
 * when SIRIUS_EXP_ONPAIR_DEFER_PREPARE is off or @p n_columns is 0.
 */
std::unique_ptr<bool[]> onpair_deferred_preparation(decompression_pushdown_scan const* scan,
                                                    std::size_t n_columns);

/**
 * @brief Register Simpatico compression/decompression converters into @p registry.
 *
 * Registers:
 *   compressed_host_representation → gpu_table_representation
 *     (read the serialised .hpln file, select columns if projected, decompress)
 *
 * Called from converter_registry::initialize().
 */
void register_compression_converters(cucascade::representation_converter_registry& registry);

}  // namespace sirius
