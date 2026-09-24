//! Bounded least-recently-used cache for prefilled session prefixes.
//!
//! The dominant cost of a request is the forward pass over the shared prefix:
//! the system context plus the per-request `state`. The context is prefilled
//! once at startup, but the state is recomputed on every request even when the
//! same state reappears (for example many questions bundled by a caller, or
//! repeated evaluations over one document).
//!
//! This cache stores the prefilled key/value cache of `system + state`, keyed
//! by a canonical hash of the state, so a reappearing state skips that forward
//! pass entirely. It is bounded both by a maximum number of entries and by the
//! total number of tokens held across all entries, evicting the least recently
//! used entry first.
//!
//! Values are stored by value and only ever handed out through [`lookup`],
//! which returns a reference: every request must clone the cached value before
//! using it, preserving the invariant that the cached cache is never mutated.
//!
//! [`lookup`]: SessionPrefixCache::lookup

use std::collections::HashMap;

/// Default number of distinct states retained by the session prefix cache.
pub const DEFAULT_SESSION_CACHE_ENTRIES: usize = 16;
/// Default total token budget across all retained session prefixes.
pub const DEFAULT_SESSION_CACHE_TOKENS: usize = 32768;

/// Bounds of the session prefix cache (entry count and total tokens).
#[derive(Debug, Clone, Copy)]
pub struct SessionCacheConfiguration {
    pub maximum_entries: usize,
    pub maximum_tokens: usize,
}

impl Default for SessionCacheConfiguration {
    fn default() -> Self {
        Self {
            maximum_entries: DEFAULT_SESSION_CACHE_ENTRIES,
            maximum_tokens: DEFAULT_SESSION_CACHE_TOKENS,
        }
    }
}

/// One cached value together with its token footprint and recency stamp.
#[derive(Debug, Clone)]
struct CacheEntry<Value> {
    value: Value,
    token_count: usize,
    last_used: u64,
}

/// Bounded least-recently-used cache keyed by a `u64` hash.
#[derive(Debug)]
pub struct SessionPrefixCache<Value> {
    entries: HashMap<u64, CacheEntry<Value>>,
    maximum_entries: usize,
    maximum_tokens: usize,
    total_tokens: usize,
    clock: u64,
}

impl<Value> SessionPrefixCache<Value> {
    /// Builds an empty cache bounded by entry count and total token count.
    ///
    /// A zero bound disables retention for that dimension: with
    /// `maximum_entries = 0` every insertion is immediately evicted, so the
    /// cache behaves as disabled.
    pub fn new(maximum_entries: usize, maximum_tokens: usize) -> Self {
        Self {
            entries: HashMap::new(),
            maximum_entries,
            maximum_tokens,
            total_tokens: 0,
            clock: 0,
        }
    }

    /// Returns the cached value for `key`, marking it as most recently used.
    pub fn lookup(&mut self, key: u64) -> Option<&Value> {
        self.clock = self.clock.wrapping_add(1);
        let clock = self.clock;
        let entry = self.entries.get_mut(&key)?;
        entry.last_used = clock;
        Some(&entry.value)
    }

    /// Stores `value` under `key`, evicting least recently used entries until
    /// both bounds hold again.
    pub fn insert(&mut self, key: u64, value: Value, token_count: usize) {
        self.clock = self.clock.wrapping_add(1);
        if let Some(previous) = self.entries.remove(&key) {
            self.total_tokens = self.total_tokens.saturating_sub(previous.token_count);
        }
        self.entries.insert(
            key,
            CacheEntry {
                value,
                token_count,
                last_used: self.clock,
            },
        );
        self.total_tokens = self.total_tokens.saturating_add(token_count);
        self.evict_to_limits();
    }

    /// Removes least recently used entries until both bounds are satisfied.
    fn evict_to_limits(&mut self) {
        while self.entries.len() > self.maximum_entries || self.total_tokens > self.maximum_tokens {
            let Some(oldest_key) = self.least_recently_used_key() else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest_key) {
                self.total_tokens = self.total_tokens.saturating_sub(removed.token_count);
            }
        }
    }

    /// Key of the entry with the smallest recency stamp, if any.
    fn least_recently_used_key(&self) -> Option<u64> {
        self.entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| *key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIVE_ENTRIES: usize = 5;
    const UNBOUNDED_TOKENS: usize = usize::MAX;

    #[test]
    fn insertion_and_lookup_round_trip() {
        let mut cache: SessionPrefixCache<String> =
            SessionPrefixCache::new(FIVE_ENTRIES, UNBOUNDED_TOKENS);
        assert!(cache.entries.is_empty());
        cache.insert(7, "seven".to_string(), 3);
        assert_eq!(cache.lookup(7).map(String::as_str), Some("seven"));
        assert!(cache.lookup(8).is_none());
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.total_tokens, 3);
    }

    #[test]
    fn replacement_replaces_value_and_token_footprint() {
        let mut cache: SessionPrefixCache<String> =
            SessionPrefixCache::new(FIVE_ENTRIES, UNBOUNDED_TOKENS);
        cache.insert(1, "first".to_string(), 10);
        cache.insert(1, "second".to_string(), 4);
        assert_eq!(cache.lookup(1).map(String::as_str), Some("second"));
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.total_tokens, 4);
    }

    #[test]
    fn entry_bound_evicts_least_recently_used() {
        let mut cache: SessionPrefixCache<u32> = SessionPrefixCache::new(2, UNBOUNDED_TOKENS);
        cache.insert(1, 10, 1);
        cache.insert(2, 20, 1);
        // Touch 1 so 2 becomes the least recently used.
        assert_eq!(cache.lookup(1), Some(&10));
        cache.insert(3, 30, 1);
        assert_eq!(cache.entries.len(), 2);
        assert!(cache.lookup(2).is_none());
        assert_eq!(cache.lookup(1), Some(&10));
        assert_eq!(cache.lookup(3), Some(&30));
    }

    #[test]
    fn token_bound_evicts_least_recently_used() {
        let mut cache: SessionPrefixCache<u32> = SessionPrefixCache::new(10, 5);
        cache.insert(1, 10, 3);
        cache.insert(2, 20, 3);
        // Inserting 3 tokens over the 5-token budget must evict key 1.
        cache.insert(3, 30, 3);
        assert_eq!(cache.total_tokens, 3);
        assert!(cache.lookup(1).is_none());
        assert_eq!(cache.lookup(3), Some(&30));
    }

    #[test]
    fn zero_entry_bound_disables_retention() {
        let mut cache: SessionPrefixCache<u32> = SessionPrefixCache::new(0, UNBOUNDED_TOKENS);
        cache.insert(1, 10, 1);
        assert!(cache.entries.is_empty());
        assert!(cache.lookup(1).is_none());
    }

    #[test]
    fn oversized_entry_is_not_retained_beyond_token_budget() {
        let mut cache: SessionPrefixCache<u32> = SessionPrefixCache::new(10, 5);
        cache.insert(1, 10, 100);
        assert!(cache.entries.is_empty());
        assert_eq!(cache.total_tokens, 0);
    }
}
