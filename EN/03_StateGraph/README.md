# 03_StateGraph

The same examples as 01_LangChain, rewritten with StateGraph.

Control flow is declared as a combination of state, nodes, and edges.

Prompts and chains are the same as in 01_LangChain.

Only the way the control flow is written differs.

Compare files with the same number in 01_LangChain and 03_StateGraph to see the differences.

## Graph API concepts

| Component | Role |
|---|---|
| State | Data passed throughout the graph. Define fields with `TypedDict`. `MessagesState` holds conversation history (messages) |
| Node | One processing step. Receives state and returns a dictionary containing only the fields to update |
| Edge (`add_edge`) | The next node to execute. Edges from one node to multiple nodes cause concurrent execution |
| Conditional edge (`add_conditional_edges`) | Selects the next node based on state. Used for branching and loops |
| `compile()` | Makes the graph executable. Pass `checkpointer` and `store` here as well |

