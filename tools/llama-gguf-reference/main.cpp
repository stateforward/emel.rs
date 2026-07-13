#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint64_t k_fnv_offset = UINT64_C(0xcbf29ce484222325);
constexpr uint64_t k_fnv_prime = UINT64_C(0x00000100000001b3);

struct contexts {
  gguf_context * gguf = nullptr;
  ggml_context * ggml = nullptr;

  ~contexts() {
    if (ggml != nullptr) {
      ggml_free(ggml);
    }
    if (gguf != nullptr) {
      gguf_free(gguf);
    }
  }
};

size_t scalar_size(const gguf_type type) {
  switch (type) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_BOOL: return 1;
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16: return 2;
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32: return 4;
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64: return 8;
    case GGUF_TYPE_STRING:
    case GGUF_TYPE_ARRAY:
    case GGUF_TYPE_COUNT: return 0;
  }
  return 0;
}

void append_u32(std::vector<uint8_t> & bytes, const uint32_t value) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    bytes.push_back(static_cast<uint8_t>(value >> (index * 8U)));
  }
}

void append_u64(std::vector<uint8_t> & bytes, const uint64_t value) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    bytes.push_back(static_cast<uint8_t>(value >> (index * 8U)));
  }
}

void append_string(std::vector<uint8_t> & bytes, const char * value) {
  const size_t length = std::strlen(value);
  append_u64(bytes, static_cast<uint64_t>(length));
  bytes.insert(bytes.end(), value, value + length);
}

std::vector<uint8_t> canonical_value(const gguf_context * context, const int64_t index) {
  std::vector<uint8_t> bytes;
  const gguf_type value_type = gguf_get_kv_type(context, index);
  if (value_type == GGUF_TYPE_STRING) {
    append_string(bytes, gguf_get_val_str(context, index));
    return bytes;
  }
  if (value_type != GGUF_TYPE_ARRAY) {
    const size_t size = scalar_size(value_type);
    const auto * data = static_cast<const uint8_t *>(gguf_get_val_data(context, index));
    bytes.insert(bytes.end(), data, data + size);
    return bytes;
  }

  const gguf_type element_type = gguf_get_arr_type(context, index);
  const size_t count = gguf_get_arr_n(context, index);
  append_u32(bytes, static_cast<uint32_t>(element_type));
  append_u64(bytes, static_cast<uint64_t>(count));
  if (element_type == GGUF_TYPE_STRING) {
    for (size_t element = 0; element < count; ++element) {
      append_string(bytes, gguf_get_arr_str(context, index, element));
    }
    return bytes;
  }
  const size_t size = scalar_size(element_type);
  const auto * data = static_cast<const uint8_t *>(gguf_get_arr_data(context, index));
  bytes.insert(bytes.end(), data, data + count * size);
  return bytes;
}

void print_hex(const uint8_t * bytes, const size_t size) {
  const auto flags = std::cout.flags();
  const char fill = std::cout.fill();
  std::cout << std::hex << std::setfill('0');
  for (size_t index = 0; index < size; ++index) {
    std::cout << std::setw(2) << static_cast<unsigned int>(bytes[index]);
  }
  std::cout.flags(flags);
  std::cout.fill(fill);
}

uint64_t fnv1a64(const uint8_t * bytes, const size_t size) {
  uint64_t hash = k_fnv_offset;
  for (size_t index = 0; index < size; ++index) {
    hash = (hash ^ bytes[index]) * k_fnv_prime;
  }
  return hash;
}

bool add_size(const size_t left, const size_t right, size_t & output) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

bool padded_size(const size_t value, const size_t alignment, size_t & output) {
  const size_t remainder = value % alignment;
  return add_size(value, remainder == 0 ? 0 : alignment - remainder, output);
}

bool tensor_data_is_present(const gguf_context * context, const size_t file_size) {
  const int64_t count = gguf_get_n_tensors(context);
  if (count == 0) {
    return true;
  }
  const size_t alignment = gguf_get_alignment(context);
  const size_t data_offset = gguf_get_data_offset(context);
  size_t required = data_offset;
  for (int64_t index = 0; index < count; ++index) {
    size_t padded = 0;
    size_t relative_end = 0;
    size_t absolute_end = 0;
    if (!padded_size(gguf_get_tensor_size(context, index), alignment, padded) ||
        !add_size(gguf_get_tensor_offset(context, index), padded, relative_end) ||
        !add_size(data_offset, relative_end, absolute_end)) {
      return false;
    }
    required = std::max(required, absolute_end);
  }
  return required <= file_size;
}

std::vector<uint8_t> read_file(const char * path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

int run(const char * path) {
  const std::vector<uint8_t> file = read_file(path);
  std::cout << "gguf-parity/v1\n";
  if (file.empty()) {
    std::cout << "status=error\n";
    return 0;
  }

  contexts owner;
  const gguf_init_params parameters = {
      /* .no_alloc = */ true,
      /* .ctx = */ &owner.ggml,
  };
  owner.gguf = gguf_init_from_file(path, parameters);
  if (owner.gguf == nullptr || !tensor_data_is_present(owner.gguf, file.size())) {
    std::cout << "status=error\n";
    return 0;
  }

  std::cout << "status=ok\n";
  std::cout << "version=" << gguf_get_version(owner.gguf) << '\n';
  std::cout << "alignment=" << gguf_get_alignment(owner.gguf) << '\n';
  std::cout << "kv_count=" << gguf_get_n_kv(owner.gguf) << '\n';
  std::cout << "tensor_count=" << gguf_get_n_tensors(owner.gguf) << '\n';

  for (int64_t index = 0; index < gguf_get_n_kv(owner.gguf); ++index) {
    const char * key = gguf_get_key(owner.gguf, index);
    std::cout << "kv." << index << ".key=";
    print_hex(reinterpret_cast<const uint8_t *>(key), std::strlen(key));
    std::cout << '\n';
    std::cout << "kv." << index << ".type=" << gguf_get_kv_type(owner.gguf, index) << '\n';
    const std::vector<uint8_t> value = canonical_value(owner.gguf, index);
    std::cout << "kv." << index << ".value=";
    print_hex(value.data(), value.size());
    std::cout << '\n';
  }

  const size_t data_offset = gguf_get_data_offset(owner.gguf);
  for (int64_t index = 0; index < gguf_get_n_tensors(owner.gguf); ++index) {
    const char * name = gguf_get_tensor_name(owner.gguf, index);
    const ggml_tensor * tensor = ggml_get_tensor(owner.ggml, name);
    if (tensor == nullptr) {
      return 2;
    }
    std::cout << "tensor." << index << ".name=";
    print_hex(reinterpret_cast<const uint8_t *>(name), std::strlen(name));
    std::cout << '\n';
    std::cout << "tensor." << index << ".type=" << gguf_get_tensor_type(owner.gguf, index)
              << '\n';
    std::cout << "tensor." << index << ".dims=" << tensor->ne[0] << ',' << tensor->ne[1]
              << ',' << tensor->ne[2] << ',' << tensor->ne[3] << '\n';
    const size_t relative_offset = gguf_get_tensor_offset(owner.gguf, index);
    const size_t size = gguf_get_tensor_size(owner.gguf, index);
    std::cout << "tensor." << index << ".offset=" << relative_offset << '\n';
    std::cout << "tensor." << index << ".size=" << size << '\n';
    const size_t absolute_offset = data_offset + relative_offset;
    const uint64_t hash = fnv1a64(file.data() + absolute_offset, size);
    const auto flags = std::cout.flags();
    const char fill = std::cout.fill();
    std::cout << "tensor." << index << ".hash=" << std::hex << std::setfill('0')
              << std::setw(16) << hash << '\n';
    std::cout.flags(flags);
    std::cout.fill(fill);
  }
  return 0;
}

}  // namespace

int main(const int argc, char ** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: llama-gguf-reference <model.gguf>\n");
    return 2;
  }
  return run(argv[1]);
}
