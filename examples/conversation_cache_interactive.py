"""
Interactive agent to test conversation string caching.
Run: python examples/conversation_cache_interactive.py

HOW THE CACHE WORKS
-------------------
get_str() returns a cached string if the conversation history hasn't changed
since the last call. The cache is invalidated any time a message is added,
updated, or deleted.

WHEN YOU SEE MISSES
-------------------
Inside agent.run(), every get_str() call that builds the LLM prompt is
preceded by an add() that appends the latest message. So each internal call
is a miss — the history just changed, the string must be rebuilt. This is
correct behavior.

WHEN YOU SEE HITS
-----------------
Hits occur when the same unchanged history is read multiple times — the
primary use case is a swarm orchestrator injecting one agent's context into
N downstream agents. Without the cache, that's N string rebuilds from the
full message list. With the cache, it's 1 rebuild and N-1 lookups.

This example simulates that pattern: after each agent response, the final
history is dispatched to NUM_WORKERS downstream agents. Each dispatch reads
the history via get_str() — the first is a miss (history was just updated by
run()), and the remaining N-1 are hits.
"""

from swarms import Agent

NUM_WORKERS = 4


def main() -> None:
    agent = Agent(
        agent_name="CacheTestAgent",
        model_name="claude-sonnet-4-5",
        max_loops=1,
        verbose=False,
        temperature=1.0,
    )

    print("\n=== Conversation Cache Interactive Test ===")
    print(
        f"After each response, history is dispatched to {NUM_WORKERS} downstream workers."
    )
    print(
        "  Misses = string rebuilt  (history changed between calls)"
    )
    print(
        "  Hits   = string reused   (history unchanged between calls)"
    )
    print("Type 'quit' to exit.\n")

    turn = 0
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        turn += 1
        hits_before = agent.short_memory._cache_hits

        response = agent.run(user_input)
        print(f"\nAgent: {response}\n")

        # Simulate dispatching the final history to N downstream workers.
        # In a real swarm, each worker would prepend this as its system context.
        # First call: miss (run() just added the response, invalidating cache).
        # Calls 2..N: hits (history unchanged between dispatches).
        for i in range(NUM_WORKERS):
            context = agent.short_memory.get_str()
            _ = context  # worker would use this as its context window

        stats = agent.short_memory.get_cache_stats()
        hits_this_turn = stats["hits"] - hits_before
        tokens_saved_this_turn = hits_this_turn * stats["cached_tokens"]

        print(f"--- Turn {turn} ---")
        print(f"  Cached tokens (history size) : {stats['cached_tokens']}")
        print(f"  Hits this turn               : {hits_this_turn}  (workers 2..{NUM_WORKERS} reused string)")
        print(f"  Tokens saved this turn       : {tokens_saved_this_turn}  ({hits_this_turn} hits x {stats['cached_tokens']} tokens)")
        print(f"  Cumulative — hits: {stats['hits']}  misses: {stats['misses']}  hit rate: {stats['hit_rate']:.0%}")
        print("------------------\n")


if __name__ == "__main__":
    main()
