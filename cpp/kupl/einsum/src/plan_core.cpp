/* Greedy contraction-path planning kernel.
 *
 * ``GreedySsa`` mirrors opt_einsum 3.4 ``opt_einsum/paths.py``: merge duplicate
 * operands eagerly (Hadamard), then greedily contract pairs whose cost is
 * ``size12 - size1 - size2`` with ties broken by ``(cost, id2, id1)``, then
 * greedily multiply out the disconnected remainder. Symbol sets are bitsets, so
 * every set operation stays integer work over ``num_symbols`` bits and no
 * hashing of Python-style index names is needed.
 *
 * Index sizes are multiplied through ``__int128`` and saturated to INT64_MAX so
 * a hostile size table can never overflow the int64 wire format.
 */

#include "einsum_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pyqcd_einsum {
namespace {

using i128 = __int128_t;
using Words = std::vector<uint64_t>;

constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
constexpr int64_t kInt64Min = std::numeric_limits<int64_t>::min();
constexpr int32_t kMaxTensors = 1 << 16;
constexpr int64_t kKeyWordBudget = int64_t(1) << 24;

inline int64_t saturate(i128 value) {
    if (value > static_cast<i128>(kInt64Max)) {
        return kInt64Max;
    }
    if (value < static_cast<i128>(kInt64Min)) {
        return kInt64Min;
    }
    return static_cast<int64_t>(value);
}

inline bool words_empty(const Words& words) {
    for (uint64_t word : words) {
        if (word != 0) {
            return false;
        }
    }
    return true;
}

/* Product of the sizes of every symbol present in ``words``; saturating. */
int64_t words_product(const Words& words, const int64_t* sizes) {
    i128 total = 1;
    for (std::size_t index = 0; index < words.size(); ++index) {
        uint64_t word = words[index];
        while (word != 0) {
            const int32_t symbol = static_cast<int32_t>(index * 64) + __builtin_ctzll(word);
            word &= word - 1;
            total *= static_cast<i128>(sizes[symbol]);
            if (total > static_cast<i128>(kInt64Max)) {
                return kInt64Max;
            }
        }
    }
    return static_cast<int64_t>(total);
}

struct WordHash {
    std::size_t operator()(const Words& words) const noexcept {
        uint64_t hash = 0xcbf29ce484222325ULL;
        for (uint64_t word : words) {
            hash ^= word;
            hash *= 0x100000001b3ULL;
            hash ^= hash >> 29;
        }
        return static_cast<std::size_t>(hash);
    }
};

/* Interning table of symbol sets: a key id is a compact, comparable set handle. */
class KeyTable {
  public:
    KeyTable(int32_t num_symbols, int64_t word_budget)
        : num_words_((static_cast<int64_t>(num_symbols) + 63) / 64), word_budget_(word_budget) {}

    int64_t num_words() const { return num_words_; }

    Words make_words() const { return Words(static_cast<std::size_t>(num_words_), 0); }

    int32_t intern(const int32_t* symbols, int32_t count) {
        Words words = make_words();
        for (int32_t index = 0; index < count; ++index) {
            const int32_t symbol = symbols[index];
            words[static_cast<std::size_t>(symbol >> 6)] |= uint64_t(1) << (symbol & 63);
        }
        return intern_words(std::move(words));
    }

    int32_t intern_words(Words&& words) {
        const auto found = index_.find(words);
        if (found != index_.end()) {
            return found->second;
        }
        if (static_cast<int64_t>(keys_.size() + 1) * num_words_ > word_budget_) {
            throw PlanBudgetExceeded();
        }
        const int32_t key = static_cast<int32_t>(keys_.size());
        keys_.push_back(std::move(words));
        index_.emplace(keys_.back(), key);
        return key;
    }

    const Words& words(int32_t key) const { return keys_[static_cast<std::size_t>(key)]; }

    bool test(int32_t key, int32_t symbol) const {
        return ((words(key)[static_cast<std::size_t>(symbol >> 6)] >> (symbol & 63)) & uint64_t(1)) != 0;
    }

    int64_t product(int32_t key, const int64_t* sizes) const { return words_product(words(key), sizes); }

    int64_t masked_product(int32_t key, int32_t mask_key, const int64_t* sizes) const {
        const Words& source = words(key);
        const Words& mask = words(mask_key);
        Words overlap = Words(source.size(), 0);
        for (std::size_t index = 0; index < source.size(); ++index) {
            overlap[index] = source[index] & mask[index];
        }
        return words_product(overlap, sizes);
    }

    /* (lhs | rhs) & mask_key */
    int32_t unite_masked(int32_t lhs, int32_t rhs, int32_t mask_key) {
        const Words& left = words(lhs);
        const Words& right = words(rhs);
        const Words& mask = words(mask_key);
        Words merged = make_words();
        for (std::size_t index = 0; index < merged.size(); ++index) {
            merged[index] = (left[index] | right[index]) & mask[index];
        }
        return intern_words(std::move(merged));
    }

    template <typename Function>
    void for_each_symbol(int32_t key, Function&& function) const {
        const Words& symbols = words(key);
        for (std::size_t index = 0; index < symbols.size(); ++index) {
            uint64_t word = symbols[index];
            while (word != 0) {
                const int32_t symbol = static_cast<int32_t>(index * 64) + __builtin_ctzll(word);
                word &= word - 1;
                function(symbol);
            }
        }
    }

    template <typename Function>
    void for_each_symbol_excluding(int32_t key, int32_t excluded_key, Function&& function) const {
        const Words& symbols = words(key);
        const Words& excluded = words(excluded_key);
        for (std::size_t index = 0; index < symbols.size(); ++index) {
            uint64_t word = symbols[index] & ~excluded[index];
            while (word != 0) {
                const int32_t symbol = static_cast<int32_t>(index * 64) + __builtin_ctzll(word);
                word &= word - 1;
                function(symbol);
            }
        }
    }

  private:
    int64_t num_words_;
    int64_t word_budget_;
    std::vector<Words> keys_;
    std::unordered_map<Words, int32_t, WordHash> index_;
};

/* Insertion-ordered key -> ssa id map, matching Python ``dict`` semantics. */
class RemainingTable {
  public:
    bool contains(int32_t key) const { return slots_.find(key) != slots_.end(); }

    int64_t get(int32_t key) const { return entries_[static_cast<std::size_t>(slots_.at(key))].ssa_id; }

    void put(int32_t key, int64_t ssa_id) {
        const auto found = slots_.find(key);
        if (found != slots_.end()) {
            entries_[static_cast<std::size_t>(found->second)].ssa_id = ssa_id;
            return;
        }
        slots_.emplace(key, static_cast<int32_t>(entries_.size()));
        entries_.push_back(Entry{key, ssa_id, true});
    }

    int64_t erase(int32_t key) {
        const auto found = slots_.find(key);
        const int32_t slot = found->second;
        const int64_t ssa_id = entries_[static_cast<std::size_t>(slot)].ssa_id;
        entries_[static_cast<std::size_t>(slot)].alive = false;
        slots_.erase(found);
        return ssa_id;
    }

    template <typename Function>
    void for_each_alive(Function&& function) const {
        for (const Entry& entry : entries_) {
            if (entry.alive) {
                function(entry.key, entry.ssa_id);
            }
        }
    }

  private:
    struct Entry {
        int32_t key;
        int64_t ssa_id;
        bool alive;
    };
    std::vector<Entry> entries_;
    std::unordered_map<int32_t, int32_t> slots_;
};

struct Candidate {
    i128 cost;
    int64_t id2;
    int64_t id1;
    int32_t k1;
    int32_t k2;
    Words k12;
};

bool candidate_before(const Candidate& lhs, const Candidate& rhs) {
    if (lhs.cost != rhs.cost) {
        return lhs.cost < rhs.cost;
    }
    if (lhs.id2 != rhs.id2) {
        return lhs.id2 < rhs.id2;
    }
    if (lhs.id1 != rhs.id1) {
        return lhs.id1 < rhs.id1;
    }
    if (lhs.k1 != rhs.k1) {
        return lhs.k1 < rhs.k1;
    }
    return lhs.k2 < rhs.k2;
}

struct CandidateMinHeap {
    bool operator()(const Candidate& lhs, const Candidate& rhs) const { return candidate_before(rhs, lhs); }
};

struct OuterEntry {
    int64_t size;
    int64_t ssa_id;
    int32_t key;
    int64_t stamp;
};

struct OuterMinHeap {
    bool operator()(const OuterEntry& lhs, const OuterEntry& rhs) const {
        if (lhs.size != rhs.size) {
            return lhs.size > rhs.size;
        }
        if (lhs.ssa_id != rhs.ssa_id) {
            return lhs.ssa_id > rhs.ssa_id;
        }
        return lhs.stamp > rhs.stamp;
    }
};

/* Greedy SSA path search; see opt_einsum.paths.ssa_greedy_optimize. */
class GreedySsa {
  public:
    GreedySsa(const PlanSpec& spec, KeyTable& keys, const std::vector<int32_t>& tensors, int32_t output_key)
        : spec_(spec), keys_(keys), tensors_(tensors) {
        Words common = keys_.words(tensors_[0]);
        for (std::size_t index = 1; index < tensors_.size(); ++index) {
            const Words& other = keys_.words(tensors_[index]);
            for (std::size_t word = 0; word < common.size(); ++word) {
                common[word] &= other[word];
            }
        }
        for (std::size_t word = 0; word < common.size(); ++word) {
            common[word] |= keys_.words(output_key)[word];
        }
        aug_output_ = keys_.intern_words(std::move(common));
        mask2_ = keys_.make_words();
        mask3_ = keys_.make_words();
        dim_to_keys_.resize(static_cast<std::size_t>(spec_.num_symbols));
        next_id_ = spec_.num_tensors;
    }

    void run(std::vector<std::pair<int64_t, int64_t>>* path) {
        path_ = path;
        merge_duplicates();
        register_dims();
        seed_candidates();
        contract_pairs();
        contract_outer_products();
    }

  private:
    void merge_duplicates() {
        for (int32_t index = 0; index < spec_.num_tensors; ++index) {
            const int32_t key = tensors_[static_cast<std::size_t>(index)];
            if (remaining_.contains(key)) {
                path_->emplace_back(remaining_.get(key), index);
                remaining_.put(key, next_id_++);
            } else {
                remaining_.put(key, index);
            }
        }
    }

    void register_dims() {
        remaining_.for_each_alive([this](int32_t key, int64_t) {
            keys_.for_each_symbol_excluding(key, aug_output_, [this, key](int32_t symbol) {
                dim_to_keys_[static_cast<std::size_t>(symbol)].insert(key);
            });
            footprints_[key] = keys_.product(key, spec_.sizes);
        });
        for (int32_t symbol = 0; symbol < spec_.num_symbols; ++symbol) {
            if (keys_.test(aug_output_, symbol)) {
                continue;
            }
            const std::size_t count = dim_to_keys_[static_cast<std::size_t>(symbol)].size();
            const std::size_t slot = static_cast<std::size_t>(symbol >> 6);
            const uint64_t bit = uint64_t(1) << (symbol & 63);
            if (count >= 2) {
                mask2_[slot] |= bit;
            }
            if (count >= 3) {
                mask3_[slot] |= bit;
            }
        }
    }

    void seed_candidates() {
        std::vector<int32_t> ordered;
        for (int32_t symbol = 0; symbol < spec_.num_symbols; ++symbol) {
            const std::set<int32_t>& sharing = dim_to_keys_[static_cast<std::size_t>(symbol)];
            if (sharing.size() < 2) {
                continue;
            }
            ordered.assign(sharing.begin(), sharing.end());
            std::sort(ordered.begin(), ordered.end(),
                      [this](int32_t lhs, int32_t rhs) { return remaining_.get(lhs) < remaining_.get(rhs); });
            for (std::size_t index = 0; index + 1 < ordered.size(); ++index) {
                push_candidate(ordered[index], ordered, index + 1);
            }
        }
    }

    void contract_pairs() {
        while (!queue_.empty()) {
            const Candidate chosen = queue_.top();
            queue_.pop();
            if (!remaining_.contains(chosen.k1) || !remaining_.contains(chosen.k2)) {
                continue; /* stale candidate */
            }
            const int64_t ssa_id1 = remaining_.erase(chosen.k1);
            const int64_t ssa_id2 = remaining_.erase(chosen.k2);

            keys_.for_each_symbol_excluding(chosen.k1, aug_output_, [this, &chosen](int32_t symbol) {
                dim_to_keys_[static_cast<std::size_t>(symbol)].erase(chosen.k1);
            });
            keys_.for_each_symbol_excluding(chosen.k2, aug_output_, [this, &chosen](int32_t symbol) {
                dim_to_keys_[static_cast<std::size_t>(symbol)].erase(chosen.k2);
            });

            path_->emplace_back(ssa_id1, ssa_id2);
            const int32_t k12 = keys_.intern_words(Words(chosen.k12));
            if (remaining_.contains(k12)) {
                path_->emplace_back(remaining_.get(k12), next_id_++);
            } else {
                keys_.for_each_symbol_excluding(k12, aug_output_, [this, k12](int32_t symbol) {
                    dim_to_keys_[static_cast<std::size_t>(symbol)].insert(k12);
                });
            }
            remaining_.put(k12, next_id_++);
            refresh_ref_counts(chosen.k1, chosen.k2);
            footprints_[k12] = keys_.product(k12, spec_.sizes);

            std::set<int32_t> neighbours;
            keys_.for_each_symbol(k12, [this, &neighbours](int32_t symbol) {
                for (int32_t other : dim_to_keys_[static_cast<std::size_t>(symbol)]) {
                    neighbours.insert(other);
                }
            });
            neighbours.erase(k12);
            if (!neighbours.empty()) {
                neighbour_scratch_.assign(neighbours.begin(), neighbours.end());
                push_candidate(k12, neighbour_scratch_, 0);
            }
        }
    }

    /* opt_einsum refreshes the dims of ``k1 | k2 - output``. */
    void refresh_ref_counts(int32_t k1, int32_t k2) {
        auto refresh = [this](int32_t symbol) {
            const std::size_t count = dim_to_keys_[static_cast<std::size_t>(symbol)].size();
            const std::size_t slot = static_cast<std::size_t>(symbol >> 6);
            const uint64_t bit = uint64_t(1) << (symbol & 63);
            if (keys_.test(aug_output_, symbol) || count <= 1) {
                mask2_[slot] &= ~bit;
                mask3_[slot] &= ~bit;
            } else if (count == 2) {
                mask2_[slot] |= bit;
                mask3_[slot] &= ~bit;
            } else {
                mask2_[slot] |= bit;
                mask3_[slot] |= bit;
            }
        };
        keys_.for_each_symbol(k1, refresh);
        keys_.for_each_symbol_excluding(k2, aug_output_, refresh);
    }

    /* opt_einsum._get_candidate */
    Candidate make_candidate(int32_t k1, int32_t k2) const {
        const Words& left = keys_.words(k1);
        const Words& right = keys_.words(k2);
        const Words& exposed = keys_.words(aug_output_);
        Words k12 = keys_.make_words();
        for (std::size_t index = 0; index < k12.size(); ++index) {
            const uint64_t either = left[index] | right[index];
            const uint64_t two = left[index] & right[index];
            const uint64_t one = either & ~two;
            k12[index] = (either & exposed[index]) | (two & mask3_[index]) | (one & mask2_[index]);
        }

        const i128 size12 = static_cast<i128>(words_product(k12, spec_.sizes));
        const i128 size1 = static_cast<i128>(footprint(k1));
        const i128 size2 = static_cast<i128>(footprint(k2));
        int64_t id1 = remaining_.get(k1);
        int64_t id2 = remaining_.get(k2);
        int32_t first = k1;
        int32_t second = k2;
        if (id1 > id2) {
            std::swap(first, second);
            std::swap(id1, id2);
        }
        return Candidate{size12 - size1 - size2, id2, id1, first, second, std::move(k12)};
    }

    void push_candidate(int32_t k1, const std::vector<int32_t>& k2s, std::size_t offset) {
        bool have_best = false;
        Candidate best{};
        for (std::size_t index = offset; index < k2s.size(); ++index) {
            Candidate candidate = make_candidate(k1, k2s[index]);
            if (!have_best || candidate_before(candidate, best)) {
                best = std::move(candidate);
                have_best = true;
            }
        }
        if (have_best) {
            queue_.push(std::move(best));
        }
    }

    void contract_outer_products() {
        std::priority_queue<OuterEntry, std::vector<OuterEntry>, OuterMinHeap> queue;
        int64_t stamp = 0;
        remaining_.for_each_alive([this, &queue, &stamp](int32_t key, int64_t ssa_id) {
            queue.push(OuterEntry{keys_.masked_product(key, aug_output_, spec_.sizes), ssa_id, key, stamp++});
        });
        if (queue.empty()) {
            return;
        }
        int64_t ssa_id1 = queue.top().ssa_id;
        int32_t k1 = queue.top().key;
        queue.pop();
        while (!queue.empty()) {
            const int64_t ssa_id2 = queue.top().ssa_id;
            const int32_t k2 = queue.top().key;
            queue.pop();
            path_->emplace_back(std::min(ssa_id1, ssa_id2), std::max(ssa_id1, ssa_id2));
            const int32_t k12 = keys_.unite_masked(k1, k2, aug_output_);
            queue.push(OuterEntry{keys_.product(k12, spec_.sizes), next_id_++, k12, stamp++});
            ssa_id1 = queue.top().ssa_id;
            k1 = queue.top().key;
            queue.pop();
        }
    }

    int64_t footprint(int32_t key) const { return footprints_.at(key); }

    const PlanSpec& spec_;
    KeyTable& keys_;
    const std::vector<int32_t>& tensors_;
    int32_t aug_output_ = -1;
    Words mask2_;
    Words mask3_;
    std::vector<std::set<int32_t>> dim_to_keys_;
    std::unordered_map<int32_t, int64_t> footprints_;
    RemainingTable remaining_;
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateMinHeap> queue_;
    std::vector<std::pair<int64_t, int64_t>>* path_ = nullptr;
    std::vector<int32_t> neighbour_scratch_;
    int64_t next_id_ = 0;
};

struct IndexStep {
    int32_t first;
    int32_t second; /* negative marks a unary step */
};

/* opt_einsum.paths.ssa_to_linear */
std::vector<IndexStep> linearize(const std::vector<std::pair<int64_t, int64_t>>& ssa_path) {
    std::vector<IndexStep> path;
    path.reserve(ssa_path.size());
    /* every ssa step is binary: sum(len) - len + 1 == len + 1 */
    std::vector<int64_t> live(ssa_path.size() + 1);
    for (std::size_t index = 0; index < live.size(); ++index) {
        live[index] = static_cast<int64_t>(index);
    }
    int64_t ssa = static_cast<int64_t>(live.size());
    for (const auto& step : ssa_path) {
        const int64_t first =
            static_cast<int64_t>(std::lower_bound(live.begin(), live.end(), step.first) - live.begin());
        const int64_t second =
            static_cast<int64_t>(std::lower_bound(live.begin(), live.end(), step.second) - live.begin());
        const int64_t high = std::max(first, second);
        const int64_t low = std::min(first, second);
        live.erase(live.begin() + high);
        live.erase(live.begin() + low);
        live.push_back(ssa++);
        path.push_back(IndexStep{static_cast<int32_t>(low), static_cast<int32_t>(high)});
    }
    return path;
}

/* Replay the linear path exactly like opt_einsum's find_contraction loop. */
void replay_cost(const PlanSpec& spec, KeyTable& keys, const std::vector<int32_t>& tensor_keys, int32_t output_key,
                 const std::vector<IndexStep>& steps, PlanOutcome* outcome) {
    const std::size_t num_words = static_cast<std::size_t>(keys.num_words());
    std::vector<Words> current;
    current.reserve(tensor_keys.size());
    for (int32_t key : tensor_keys) {
        current.push_back(keys.words(key));
    }

    i128 total_cost = 0;
    int64_t largest = 0;
    bool have_size = false;
    for (const IndexStep& step : steps) {
        Words contract(num_words, 0);
        Words remain = keys.words(output_key);
        for (std::size_t index = 0; index < current.size(); ++index) {
            const bool contracted =
                static_cast<int32_t>(index) == step.first || static_cast<int32_t>(index) == step.second;
            for (std::size_t word = 0; word < num_words; ++word) {
                if (contracted) {
                    contract[word] |= current[index][word];
                } else {
                    remain[word] |= current[index][word];
                }
            }
        }
        Words result(num_words, 0);
        Words removed(num_words, 0);
        for (std::size_t word = 0; word < num_words; ++word) {
            result[word] = remain[word] & contract[word];
            removed[word] = contract[word] & ~result[word];
        }

        /* flop_count(idx_contract, inner, num_terms=2) */
        total_cost += static_cast<i128>(words_product(contract, spec.sizes)) * (words_empty(removed) ? 1 : 2);
        const int64_t size_result = words_product(result, spec.sizes);
        largest = have_size ? std::max(largest, size_result) : size_result;
        have_size = true;

        if (step.second >= 0) {
            current.erase(current.begin() + std::max(step.first, step.second));
            current.erase(current.begin() + std::min(step.first, step.second));
        } else {
            current.erase(current.begin() + step.first);
        }
        current.push_back(std::move(result));
    }

    outcome->opt_cost = saturate(total_cost);
    outcome->largest_intermediate = have_size ? largest : 1;
}

int plan_greedy(const PlanSpec& spec, PlanOutcome* outcome) {
    KeyTable keys(spec.num_symbols, kKeyWordBudget);
    std::vector<int32_t> tensor_keys(static_cast<std::size_t>(spec.num_tensors));
    for (int32_t index = 0; index < spec.num_tensors; ++index) {
        const TensorView& tensor = spec.tensors[index];
        tensor_keys[static_cast<std::size_t>(index)] = keys.intern(tensor.symbols, tensor.count);
    }
    const int32_t output_key = keys.intern(spec.output, spec.output_count);

    if (spec.num_tensors == 1) {
        /* opt_einsum short-circuits a single operand to the unary contraction. */
        const std::vector<IndexStep> steps{{0, -1}};
        replay_cost(spec, keys, tensor_keys, output_key, steps, outcome);
        return QCD_STATUS_OK;
    }

    std::vector<std::pair<int64_t, int64_t>> ssa_path;
    GreedySsa greedy(spec, keys, tensor_keys, output_key);
    greedy.run(&ssa_path);
    const std::vector<IndexStep> steps = linearize(ssa_path);
    replay_cost(spec, keys, tensor_keys, output_key, steps, outcome);

    outcome->path.reserve(steps.size() * 2);
    for (const IndexStep& step : steps) {
        outcome->path.push_back(step.first);
        outcome->path.push_back(step.second);
    }
    return QCD_STATUS_OK;
}

}  // namespace

int plan_contract(const PlanSpec& spec, PlanOutcome* outcome) {
    if (outcome == nullptr) {
        return QCD_STATUS_INVALID;
    }
    outcome->path.clear();
    outcome->opt_cost = 0;
    outcome->largest_intermediate = 1;
    if (spec.num_tensors < 0 || spec.num_tensors > kMaxTensors || spec.num_symbols < 0 ||
        spec.output_count < 0 || spec.output_count > spec.num_symbols) {
        return QCD_STATUS_INVALID;
    }
    if (spec.planner_id != QCD_PLANNER_GREEDY) {
        return QCD_STATUS_INVALID;
    }
    if (spec.memory_limited) {
        /* opt_einsum's greedy delegates to branch() whenever a limit is set. */
        return QCD_STATUS_INVALID;
    }
    if (spec.num_tensors == 0) {
        return QCD_STATUS_OK;
    }
    if (spec.tensors == nullptr || (spec.sizes == nullptr && spec.num_symbols > 0)) {
        return QCD_STATUS_INVALID;
    }
    return plan_greedy(spec, outcome);
}

}  // namespace pyqcd_einsum
