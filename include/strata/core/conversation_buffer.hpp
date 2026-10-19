#pragma once

#include "strata/core/progress.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace strata::core {

// Snapshot-owned host storage. Appending never relocates existing payloads;
// only the small segment directory can move. No sharing or copy-on-write is
// needed because a parked image and its retained active buffers have one owner.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t bytes() const {
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.capacity;
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity - segments_.back().size;
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity, size_ != 0, size_))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) { grow(n, value); }
    // The same growth with the new bytes left as they are (fresh pages stay
    // demand-zero from the OS, reused pages keep their old contents). Only for a
    // caller that overwrites [size(), n) before anything reads it - a capture
    // copies the whole mutable tail over exactly that range - so it saves the
    // zero-fill pass on every parked K/V payload.
    void resize_for_overwrite(size_t n) { grow(n, -1); }
    void pop_back() { resize(size_ - 1); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size - x, right.size - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size) { ++a; x = 0; }
            if (y == right.size) { ++b; y = 0; }
        }
        return true;
    }

private:
    // One allocation of at most segment_bytes. `size` is what the payload holds,
    // `capacity` what is reserved for the next append. Raw storage rather than a
    // vector<uint8_t> so growth can skip the zero-fill (resize_for_overwrite):
    // a vector's resize value-initialises the new elements unconditionally.
    struct Segment {
        std::unique_ptr<uint8_t[]> p;
        size_t size = 0;
        size_t capacity = 0;
        Segment() = default;
        Segment(const Segment& other) { copy_from(other); }
        Segment& operator=(const Segment& other) {
            if (this != &other) copy_from(other);
            return *this;
        }
        Segment(Segment&&) noexcept = default;
        Segment& operator=(Segment&&) noexcept = default;
        uint8_t* data() { return p.get(); }
        const uint8_t* data() const { return p.get(); }
        void reserve(size_t n) {
            if (n <= capacity) return;
            auto next = std::unique_ptr<uint8_t[]>(new uint8_t[n]);
            if (size) std::memcpy(next.get(), p.get(), size);
            p = std::move(next);
            capacity = n;
        }
        void resize(size_t n, uint8_t value) {
            if (n > capacity) reserve(n);
            if (n > size) std::memset(p.get() + size, value, n - size);
            size = n;
        }
        void copy_from(const Segment& other) {
            reserve(other.size);
            if (other.size) std::memcpy(p.get(), other.p.get(), other.size);
            size = other.size;
        }
    };
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    // The reserve is also bounded by an eighth of what the buffer held before
    // the append (at least 64 KiB): a parked conversation carries it in every
    // K/V buffer, and at up to 16 MiB each it came to 0.4-0.5 GB of a ~1 GB
    // conversation (Qwen3.8-Flash-Next), counted against the cache's budget.
    static size_t segment_capacity(size_t count, size_t previous, bool growing, size_t held) {
        if (!growing) return count;
        const size_t geometric = std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2));
        return std::max(count, std::min(geometric, std::max<size_t>(65536, held / 8)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size;
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + offset - begin, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    // value < 0: do not touch the new bytes (resize_for_overwrite).
    void grow(size_t n, int value) {
        if (n > size_) {
            const size_t held = size_;
            const bool growing = held != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity - segments_.back().size;
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                Segment& last = segments_.back();
                if (value < 0) last.size += extend;
                else last.resize(last.size + extend, uint8_t(value));
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity;
                Segment segment;
                segment.reserve(segment_capacity(count, previous, growing, held));
                if (value < 0) segment.size = count;
                else segment.resize(count, uint8_t(value));
                segments_.push_back(std::move(segment));
                size_ += count;
                if (value >= 0) progress_beat();   // #2049: zero-filling 16 MiB of fresh pages is slow when the host is paging
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size >= n) {
                size_ -= segments_.back().size;
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().size -= size_ - n;
            size_ = n;
        }
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
};

} // namespace strata::core
