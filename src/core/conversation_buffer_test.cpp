// Standalone check of ConversationBuffer's growth paths, including the
// resize_for_overwrite shortcut the park write uses to skip the zero-fill of a
// fresh K/V payload (src/core/conversation_snapshot.cpp, conversation_kv_save).
#include "strata/core/conversation_buffer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

static long fail_after = -1;
void* operator new(std::size_t n) {
    if (fail_after == 0) throw std::bad_alloc();
    if (fail_after > 0) --fail_after;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
std::vector<uint8_t> pattern(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(seed + i * 31 + (i >> 9));
    return v;
}
void fill(ConversationBuffer& b, const std::vector<uint8_t>& v, size_t at = 0) {
    check(b.visit(at, v.size(), [&](uint8_t* p, size_t n, size_t o) {
              std::memcpy(p, v.data() + o - at, n);
              return true;
          }), "write visits the whole range");
}
std::vector<uint8_t> dump(const ConversationBuffer& b) {
    std::vector<uint8_t> v(b.size());
    check(b.read(v.data(), 0, v.size()) || v.empty(), "read the whole buffer");
    return v;
}
} // namespace

int main() {
    const size_t seg = ConversationBuffer::segment_bytes;
    {
        // Failed allocation may leave partially grown storage; discard it before
        // retrying, exactly as a failed capture must not publish its image.
        int faults = 0;
        bool completed = false;
        for (long at = 0; at < 16; ++at) {
            ConversationBuffer b{1, 2, 3};
            fail_after = at;
            try {
                b.resize_for_overwrite(seg + 7);
                fail_after = -1;
                completed = true;
            } catch (const std::bad_alloc&) {
                fail_after = -1;
                ++faults;
                uint8_t prefix[3]{};
                check(b.read(prefix, 0, 3) && prefix[0] == 1 && prefix[2] == 3,
                      "allocation failure preserves the initialized prefix");
            }
            b.resize(0);
            b.resize_for_overwrite(123);
            fill(b, pattern(123, 0x22));
            check(dump(b) == pattern(123, 0x22), "retry after discarded capture is complete");
            if (completed) break;
        }
        check(completed && faults > 0, "allocation failure injection reached success");
    }
    {
        ConversationBuffer b;
        b.resize_for_overwrite(seg + 7);
        int visits = 0;
        check(!b.visit(0, b.size(), [&](uint8_t* p, size_t n, size_t) {
            ++visits;
            std::memset(p, 0x5a, n);
            return false;
        }), "transfer callback failure is propagated");
        check(visits == 1, "failed transfer stops before the next segment");
        b.resize(0);
        b.resize(12, 0x33);
        check(dump(b) == std::vector<uint8_t>(12, 0x33), "discard failed image then retry");
    }
    {
        // A capture grows to the payload size and then copies every byte of the
        // mutable tail over it: resize_for_overwrite must leave exactly what the
        // plain resize produces once the copy has run.
        for (size_t n : {size_t(1), size_t(100), seg - 1, seg, seg + 1, 3 * seg + 7}) {
            const auto payload = pattern(n, 0x40);
            ConversationBuffer plain, shortcut;
            plain.resize(n);
            shortcut.resize_for_overwrite(n);
            fill(shortcut, payload);
            check(dump(shortcut) == payload, "resize_for_overwrite range is fully writable");
            fill(plain, payload);
            check(dump(plain) == dump(shortcut), "resize and resize_for_overwrite agree after the overwrite");
        }
    }
    {
        // Recycled heap memory can still hold a previous image's bytes, so the
        // shortcut must not depend on the storage starting zeroed: the capture
        // copies over every byte of the grown range in the same call. Model that
        // by handing the allocator a dirty same-size block first.
        const size_t n = 2 * seg + 333;
        {
            ConversationBuffer dirty;
            dirty.resize(n, 0x5a);
        }
        const auto payload = pattern(n, 0x33);
        ConversationBuffer shortcut;
        shortcut.resize_for_overwrite(n);
        fill(shortcut, payload);
        check(dump(shortcut) == payload, "the capture's copy determines every grown byte");
    }
    {
        // The documented difference: growth reusing already-written space keeps
        // the old bytes instead of zeroing them. The capture copies over that
        // range in the same call, so this cannot leak into an image.
        ConversationBuffer b;
        const size_t n = 3 * seg + 7;
        b.resize(n, 0xab);
        b.resize(seg + 11);                       // shrink inside written space
        b.resize_for_overwrite(n);
        const auto bytes = dump(b);
        check(bytes.size() == n && bytes[seg + 500] == 0xab && bytes[2 * seg - 1] == 0xab,
              "resize_for_overwrite keeps reused bytes untouched");
        b.resize(seg + 11);
        b.resize(n);
        check(dump(b)[seg + 500] == 0, "resize still zero-fills the grown range");
    }
    {
        // Appending into a segment's spare capacity, then copying.
        ConversationBuffer b;
        b.resize(5 * 1024 * 1024, 1);
        const auto before = dump(b);
        const size_t grown = 20 * 1024 * 1024;
        b.resize_for_overwrite(grown);
        check(b.size() == grown, "extend-in-spare size");
        std::vector<uint8_t> head(5 * 1024 * 1024);
        check(b.read(head.data(), 0, head.size()), "read the retained prefix");
        check(head == before, "extend-in-spare keeps the prefix");
        const auto tail = pattern(grown - head.size(), 0x11);
        fill(b, tail, head.size());
        check(dump(b) == [&] {
            auto v = before;
            v.insert(v.end(), tail.begin(), tail.end());
            return v;
        }(), "extend-in-spare then write the appended tail");
    }
    {
        // Value semantics and address stability, the same guarantees the vector
        // based storage had.
        ConversationBuffer b;
        b.resize(seg + 17, 7);
        uint8_t* first = nullptr;
        b.visit(0, 1, [&](uint8_t* p, size_t, size_t) { first = p; return true; });
        const size_t peak = b.allocation_peak(seg + 116);
        b.resize(seg + 116, 9);
        b.visit(0, 1, [&](uint8_t* p, size_t, size_t) { check(p == first, "append keeps payload addresses"); return true; });
        check(b.bytes() <= peak, "growth fits the admitted peak");
        auto copy = b;
        check(dump(copy) == dump(b), "copy ctor is deep");
        const auto kept = dump(b);
        b.resize(seg + 10);
        check(dump(copy) == kept, "copy is independent of later resize");
        ConversationBuffer moved = std::move(copy);
        check(copy.empty() && copy.bytes() == 0, "move leaves the source empty");
        check(dump(moved).size() == seg + 116, "move keeps the payload");
        check(b.allocation_peak(seg + 10) >= b.bytes(), "peak covers the directory");
    }
    {
        ConversationBuffer b;
        check(b.visit(0, 1, [](uint8_t*, size_t, size_t) { return true; }) == false,
              "visit rejects a range past the end");
        b.resize(ConversationBuffer::segment_bytes, 3);
        check(b.visit(ConversationBuffer::segment_bytes, 1, [](uint8_t*, size_t, size_t) { return true; }) == false,
              "visit rejects an offset at the end");
        std::vector<uint8_t> small(ConversationBuffer::segment_bytes);
        check(b.read(small.data(), 0, small.size()), "exact-size read");
        b.pop_back();
        check(b.size() == ConversationBuffer::segment_bytes - 1, "pop_back shrinks by one");
    }
    std::printf("conversation_buffer_test: %d checks passed\n", checks);
    return 0;
}

