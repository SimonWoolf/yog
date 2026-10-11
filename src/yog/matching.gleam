//// Maximum matching in general (not necessarily bipartite) undirected graphs,
//// using [Edmonds' blossom algorithm](https://en.wikipedia.org/wiki/Blossom_algorithm).
////
//// A [matching](https://en.wikipedia.org/wiki/Matching_(graph_theory)) is a set
//// of edges no two of which share a vertex. Unlike the bipartite matching in
//// `yog/property/bipartite`, these functions handle odd cycles correctly by
//// contracting them into "blossoms".
////
//// ## Available Functions
////
//// | Function | Finds |
//// |----------|-------|
//// | `maximum_matching/1` | A matching with as many edges as possible (weights ignored) |
//// | `maximum_weight_matching/2` | A matching with the largest total weight |
//// | `max_cardinality_weight_matching/2` | The heaviest matching among those with the most edges |
////
//// Weights must be integers. This keeps every dual variable exact; for
//// fractional weights, scale them to integers first (e.g. multiply by 1000 and
//// round).
////
//// ## Algorithm
////
//// This is the primal-dual weighted blossom algorithm (Edmonds 1965, with the
//// O(V³) bookkeeping described by Galil 1986). It is a port of Joris van
//// Rantwijk's `mwmatching.py`, as adapted in NetworkX's `max_weight_matching`.
//// Persistent dictionaries add a logarithmic factor, giving O(V³ log V).
////
//// ## Use Cases
////
//// - **Pairing**: roommate assignment, pairing players or tasks where any two
////   participants can be paired (not just across two sides)
//// - **Chinese postman / T-joins**: minimum weight perfect matching on odd-degree
////   vertices (negate weights and use `max_cardinality_weight_matching/2`)
//// - **Approximation algorithms**: e.g. Christofides' algorithm for metric TSP
////
//// ## References
////
//// - Edmonds, J. (1965). "Paths, trees, and flowers"
//// - Galil, Z. (1986). "Efficient algorithms for finding maximum matching in graphs"
//// - [Van Rantwijk: Maximum Weighted Matching](https://jorisvr.nl/article/maximum-matching)

import gleam/dict.{type Dict}
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/set.{type Set}
import yog/model.{type Graph, type NodeId, Directed, Undirected}

/// Finds a maximum cardinality matching: as many vertex-disjoint edges as
/// possible. Edge data is ignored.
///
/// Returns the matched pairs `#(u, v)` with `u < v`, sorted by `u`. Returns an
/// error for directed graphs.
///
/// ## Example
///
/// ```gleam
/// // A triangle with a pendant vertex: 1-2-3-1, 3-4
/// let assert Ok(graph) =
///   yog.undirected()
///   |> yog.add_node(1, Nil)
///   |> yog.add_node(2, Nil)
///   |> yog.add_node(3, Nil)
///   |> yog.add_node(4, Nil)
///   |> yog.add_edges([#(1, 2, Nil), #(2, 3, Nil), #(3, 1, Nil), #(3, 4, Nil)])
///
/// matching.maximum_matching(graph)
/// // => Ok([#(1, 2), #(3, 4)])
/// ```
///
/// **Time Complexity:** O(V³ log V)
pub fn maximum_matching(
  graph: Graph(n, e),
) -> Result(List(#(NodeId, NodeId)), String) {
  run(graph, fn(_) { 1 }, False)
}

/// Finds a matching with the largest possible total edge weight.
///
/// Edges with non-positive weight are never worth matching, so they are never
/// included. The result is not necessarily a maximum cardinality matching;
/// use `max_cardinality_weight_matching/2` for that.
///
/// Returns the matched pairs `#(u, v)` with `u < v`, sorted by `u`. Returns an
/// error for directed graphs.
///
/// ## Example
///
/// ```gleam
/// // Path 1 -2- 2 -3- 3 -2- 4: taking the middle edge alone (3) loses to
/// // taking both outer edges (2 + 2).
/// let assert Ok(graph) =
///   yog.undirected()
///   |> yog.add_node(1, Nil)
///   |> yog.add_node(2, Nil)
///   |> yog.add_node(3, Nil)
///   |> yog.add_node(4, Nil)
///   |> yog.add_edges([#(1, 2, 2), #(2, 3, 3), #(3, 4, 2)])
///
/// matching.maximum_weight_matching(graph, fn(w) { w })
/// // => Ok([#(1, 2), #(3, 4)])
/// ```
///
/// **Time Complexity:** O(V³ log V)
pub fn maximum_weight_matching(
  graph: Graph(n, e),
  with weight: fn(e) -> Int,
) -> Result(List(#(NodeId, NodeId)), String) {
  run(graph, weight, False)
}

/// Finds the matching with the largest total weight among all matchings with
/// the maximum possible number of edges.
///
/// Negative weights are allowed. Negating all weights therefore gives a
/// minimum weight maximum cardinality matching (a minimum weight perfect
/// matching, when one exists).
///
/// Returns the matched pairs `#(u, v)` with `u < v`, sorted by `u`. Returns an
/// error for directed graphs.
///
/// ## Example
///
/// ```gleam
/// // Same path as above, but now the two-edge matching must win on
/// // cardinality even when the middle edge is heavier.
/// let assert Ok(graph) =
///   yog.undirected()
///   |> yog.add_node(1, Nil)
///   |> yog.add_node(2, Nil)
///   |> yog.add_node(3, Nil)
///   |> yog.add_node(4, Nil)
///   |> yog.add_edges([#(1, 2, 1), #(2, 3, 10), #(3, 4, 1)])
///
/// matching.max_cardinality_weight_matching(graph, fn(w) { w })
/// // => Ok([#(1, 2), #(3, 4)])
/// ```
///
/// **Time Complexity:** O(V³ log V)
pub fn max_cardinality_weight_matching(
  graph: Graph(n, e),
  with weight: fn(e) -> Int,
) -> Result(List(#(NodeId, NodeId)), String) {
  run(graph, weight, True)
}

// ============= Algorithm state =============

type Edge =
  #(NodeId, NodeId)

/// A top-level or nested blossom. Every vertex is also a trivial blossom, so
/// per-blossom data keyed by `Leaf(v)` doubles as per-vertex data, mirroring
/// the reference implementation.
type Blossom {
  Leaf(NodeId)
  Nested(Int)
}

type Label {
  SLabel
  TLabel
  /// Temporary marker used while tracing alternating paths in `scan_blossom`.
  Breadcrumb
}

/// `children` and `edges` are circular arrays (index -> item) of length `size`:
/// `edges[i]` connects a vertex in `children[i]` to one in `children[i + 1]`.
/// `children[0]` contains the base.
type NestedBlossom {
  NestedBlossom(
    children: Dict(Int, Blossom),
    edges: Dict(Int, Edge),
    size: Int,
    base: NodeId,
    /// Twice the blossom's dual variable z(b).
    dual: Int,
    /// For a top-level S-blossom, the least-slack edges to each neighbouring
    /// S-blossom; `None` until computed.
    best_edges: Option(List(Edge)),
  )
}

type Ctx {
  Ctx(
    nodes: List(NodeId),
    adj: Dict(NodeId, Dict(NodeId, Int)),
    max_cardinality: Bool,
  )
}

type State {
  State(
    mate: Dict(NodeId, NodeId),
    label: Dict(Blossom, Label),
    /// The edge through which a labelled blossom (or a reached vertex inside
    /// a T-blossom) got its label; absent if its base is single.
    label_edge: Dict(Blossom, Edge),
    in_blossom: Dict(NodeId, Blossom),
    parent: Dict(Blossom, Blossom),
    nested: Dict(Int, NestedBlossom),
    /// For a free vertex: least-slack edge from an S-vertex. For a top-level
    /// S-blossom: least-slack edge to a different S-blossom.
    best_edge: Dict(Blossom, Edge),
    /// Twice each vertex's dual variable u(v), so all arithmetic stays integral.
    dual: Dict(NodeId, Int),
    allowed: Set(Edge),
    queue: List(NodeId),
    next_id: Int,
  )
}

type Delta {
  VertexDual
  FreeVertexEdge(Edge)
  SBlossomEdge(Edge)
  ExpandTBlossom(Int)
}

fn run(
  graph: Graph(n, e),
  weight: fn(e) -> Int,
  max_cardinality: Bool,
) -> Result(List(#(NodeId, NodeId)), String) {
  case graph.kind {
    Directed -> Error("maximum matching requires an undirected graph")
    Undirected -> {
      let nodes = model.all_nodes(graph)
      let adj =
        dict.map_values(graph.out_edges, fn(u, targets) {
          targets
          |> dict.delete(u)
          |> dict.map_values(fn(_, e) { weight(e) })
        })
      let max_weight =
        dict.fold(adj, 0, fn(acc, _, targets) {
          dict.fold(targets, acc, fn(acc, _, w) { int.max(acc, w) })
        })
      let ctx = Ctx(nodes:, adj:, max_cardinality:)
      let state =
        State(
          mate: dict.new(),
          label: dict.new(),
          label_edge: dict.new(),
          in_blossom: list.fold(nodes, dict.new(), fn(acc, v) {
            dict.insert(acc, v, Leaf(v))
          }),
          parent: dict.new(),
          nested: dict.new(),
          best_edge: dict.new(),
          dual: list.fold(nodes, dict.new(), fn(acc, v) {
            dict.insert(acc, v, max_weight)
          }),
          allowed: set.new(),
          queue: [],
          next_id: 0,
        )
      let final = stage(ctx, state)
      Ok(
        dict.fold(final.mate, [], fn(acc, u, v) {
          case u < v {
            True -> [#(u, v), ..acc]
            False -> acc
          }
        })
        |> list.sort(fn(a, b) { int.compare(a.0, b.0) }),
      )
    }
  }
}

// ============= Stages =============

/// Each stage either augments the matching by one edge or proves optimality.
fn stage(ctx: Ctx, s: State) -> State {
  let s =
    State(
      ..s,
      label: dict.new(),
      label_edge: dict.new(),
      best_edge: dict.new(),
      nested: dict.map_values(s.nested, fn(_, nb) {
        NestedBlossom(..nb, best_edges: None)
      }),
      allowed: set.new(),
      queue: [],
    )
  let s =
    list.fold(ctx.nodes, s, fn(s, v) {
      case dict.has_key(s.mate, v), label_of(s, in_blossom(s, v)) {
        False, None -> assign_label(s, v, SLabel, None)
        _, _ -> s
      }
    })
  case substage(ctx, s) {
    #(s, False) -> s
    #(s, True) -> stage(ctx, expand_zero_dual_s_blossoms(s))
  }
}

fn expand_zero_dual_s_blossoms(s: State) -> State {
  list.fold(dict.keys(s.nested), s, fn(s, id) {
    case dict.get(s.nested, id) {
      Ok(nb) ->
        case
          dict.has_key(s.parent, Nested(id)),
          label_of(s, Nested(id)),
          nb.dual
        {
          False, Some(SLabel), 0 -> expand_blossom(s, id, True)
          _, _, _ -> s
        }
      Error(_) -> s
    }
  })
}

/// Grows alternating trees until an augmenting path is found (returning
/// `True`) or the duals show no further improvement is possible (`False`).
fn substage(ctx: Ctx, s: State) -> #(State, Bool) {
  case process_queue(ctx, s) {
    #(s, True) -> #(s, True)
    #(s, False) -> {
      let #(delta, kind) = compute_delta(ctx, s)
      let s = apply_delta(ctx, s, delta)
      case kind {
        VertexDual -> #(s, False)
        FreeVertexEdge(#(v, w)) | SBlossomEdge(#(v, w)) ->
          substage(ctx, State(..allow(s, v, w), queue: [v, ..s.queue]))
        ExpandTBlossom(id) -> substage(ctx, expand_blossom(s, id, False))
      }
    }
  }
}

fn process_queue(ctx: Ctx, s: State) -> #(State, Bool) {
  case s.queue {
    [] -> #(s, False)
    [v, ..rest] ->
      case
        scan_neighbours(ctx, State(..s, queue: rest), v, neighbours(ctx, v))
      {
        #(s, True) -> #(s, True)
        #(s, False) -> process_queue(ctx, s)
      }
  }
}

fn scan_neighbours(
  ctx: Ctx,
  s: State,
  v: NodeId,
  ws: List(NodeId),
) -> #(State, Bool) {
  case ws {
    [] -> #(s, False)
    [w, ..rest] -> {
      let bv = in_blossom(s, v)
      let bw = in_blossom(s, w)
      case bv == bw {
        True -> scan_neighbours(ctx, s, v, rest)
        False -> {
          let #(s, allowed, kslack) = case set.contains(s.allowed, #(v, w)) {
            True -> #(s, True, 0)
            False -> {
              let k = slack(ctx, s, v, w)
              case k <= 0 {
                True -> #(allow(s, v, w), True, k)
                False -> #(s, False, k)
              }
            }
          }
          case allowed, label_of(s, bw) {
            True, None ->
              scan_neighbours(ctx, assign_label(s, w, TLabel, Some(v)), v, rest)
            True, Some(SLabel) ->
              case scan_blossom(s, v, w) {
                Some(base) ->
                  scan_neighbours(ctx, add_blossom(ctx, s, base, v, w), v, rest)
                None -> #(augment_matching(s, v, w), True)
              }
            True, _ ->
              case label_of(s, Leaf(w)) {
                // w is inside a T-blossom but not yet reached from outside it;
                // remember how it was reached for relabelling on expansion.
                None ->
                  State(
                    ..s,
                    label: dict.insert(s.label, Leaf(w), TLabel),
                    label_edge: dict.insert(s.label_edge, Leaf(w), #(v, w)),
                  )
                  |> scan_neighbours(ctx, _, v, rest)
                Some(_) -> scan_neighbours(ctx, s, v, rest)
              }
            False, Some(SLabel) ->
              scan_neighbours(
                ctx,
                improve_best_edge(ctx, s, bv, #(v, w), kslack),
                v,
                rest,
              )
            False, _ ->
              case label_of(s, Leaf(w)) {
                None ->
                  scan_neighbours(
                    ctx,
                    improve_best_edge(ctx, s, Leaf(w), #(v, w), kslack),
                    v,
                    rest,
                  )
                Some(_) -> scan_neighbours(ctx, s, v, rest)
              }
          }
        }
      }
    }
  }
}

fn improve_best_edge(
  ctx: Ctx,
  s: State,
  key: Blossom,
  edge: Edge,
  edge_slack: Int,
) -> State {
  let better = case dict.get(s.best_edge, key) {
    Error(_) -> True
    Ok(#(a, b)) -> edge_slack < slack(ctx, s, a, b)
  }
  case better {
    True -> State(..s, best_edge: dict.insert(s.best_edge, key, edge))
    False -> s
  }
}

// ============= Dual updates =============

/// Delta values are pre-multiplied by two, like the duals.
fn compute_delta(ctx: Ctx, s: State) -> #(Int, Delta) {
  let consider = fn(best: Option(#(Int, Delta)), d: Int, kind: Delta) {
    case best {
      Some(#(current, _)) if current <= d -> best
      _ -> Some(#(d, kind))
    }
  }

  let best = case ctx.max_cardinality {
    True -> None
    False -> Some(#(min_vertex_dual(s), VertexDual))
  }

  let best =
    list.fold(ctx.nodes, best, fn(best, v) {
      case label_of(s, in_blossom(s, v)), dict.get(s.best_edge, Leaf(v)) {
        None, Ok(#(a, b) as e) ->
          consider(best, slack(ctx, s, a, b), FreeVertexEdge(e))
        _, _ -> best
      }
    })

  let top_level_s_edge = fn(best, b) {
    case dict.has_key(s.parent, b), label_of(s, b), dict.get(s.best_edge, b) {
      False, Some(SLabel), Ok(#(a, c) as e) ->
        consider(best, slack(ctx, s, a, c) / 2, SBlossomEdge(e))
      _, _, _ -> best
    }
  }
  let best =
    list.fold(ctx.nodes, best, fn(best, v) { top_level_s_edge(best, Leaf(v)) })
  let best =
    dict.fold(s.nested, best, fn(best, id, _) {
      top_level_s_edge(best, Nested(id))
    })

  let best =
    dict.fold(s.nested, best, fn(best, id, nb) {
      case dict.has_key(s.parent, Nested(id)), label_of(s, Nested(id)) {
        False, Some(TLabel) -> consider(best, nb.dual, ExpandTBlossom(id))
        _, _ -> best
      }
    })

  case best {
    Some(result) -> result
    // Only reachable with max_cardinality: no augmenting path remains. A final
    // vertex-dual step keeps the duals feasible.
    None -> #(int.max(0, min_vertex_dual(s)), VertexDual)
  }
}

fn min_vertex_dual(s: State) -> Int {
  dict.fold(s.dual, None, fn(acc, _, d) {
    case acc {
      Some(m) if m <= d -> acc
      _ -> Some(d)
    }
  })
  |> option.unwrap(0)
}

fn apply_delta(ctx: Ctx, s: State, delta: Int) -> State {
  let dual =
    list.fold(ctx.nodes, s.dual, fn(dual, v) {
      case label_of(s, in_blossom(s, v)) {
        Some(SLabel) -> dict.insert(dual, v, fetch(dual, v) - delta)
        Some(TLabel) -> dict.insert(dual, v, fetch(dual, v) + delta)
        _ -> dual
      }
    })
  let nested =
    dict.map_values(s.nested, fn(id, nb) {
      case dict.has_key(s.parent, Nested(id)), label_of(s, Nested(id)) {
        False, Some(SLabel) -> NestedBlossom(..nb, dual: nb.dual + delta)
        False, Some(TLabel) -> NestedBlossom(..nb, dual: nb.dual - delta)
        _, _ -> nb
      }
    })
  State(..s, dual:, nested:)
}

// ============= Labelling =============

/// Labels the top-level blossom containing `w` with `t`, reached via an edge
/// from `v`. A T-blossom's mate is labelled S in turn.
fn assign_label(s: State, w: NodeId, t: Label, v: Option(NodeId)) -> State {
  let b = in_blossom(s, w)
  let label_edge = case v {
    Some(v) ->
      s.label_edge
      |> dict.insert(Leaf(w), #(v, w))
      |> dict.insert(b, #(v, w))
    None ->
      s.label_edge
      |> dict.delete(Leaf(w))
      |> dict.delete(b)
  }
  let s =
    State(
      ..s,
      label: s.label |> dict.insert(Leaf(w), t) |> dict.insert(b, t),
      label_edge:,
      best_edge: s.best_edge |> dict.delete(Leaf(w)) |> dict.delete(b),
    )
  case t {
    SLabel -> State(..s, queue: list.append(leaves(s, b), s.queue))
    _ -> {
      let base = base_of(s, b)
      assign_label(s, fetch(s.mate, base), SLabel, Some(base))
    }
  }
}

/// Traces back from S-vertices `v` and `w` along their alternating trees.
/// Returns the base of the new blossom if the paths meet, or `None` if they
/// end at two different single vertices (an augmenting path).
fn scan_blossom(s: State, v: NodeId, w: NodeId) -> Option(NodeId) {
  do_scan_blossom(s, s.label, Some(v), Some(w))
}

fn do_scan_blossom(
  s: State,
  labels: Dict(Blossom, Label),
  v: Option(NodeId),
  w: Option(NodeId),
) -> Option(NodeId) {
  case v {
    None -> None
    Some(x) -> {
      let b = in_blossom(s, x)
      case dict.get(labels, b) {
        Ok(Breadcrumb) -> Some(base_of(s, b))
        _ -> {
          let labels = dict.insert(labels, b, Breadcrumb)
          let next = case dict.get(s.label_edge, b) {
            Error(_) -> None
            Ok(#(t, _)) -> Some(fetch(s.label_edge, in_blossom(s, t)).0)
          }
          case w {
            Some(_) -> do_scan_blossom(s, labels, w, next)
            None -> do_scan_blossom(s, labels, next, None)
          }
        }
      }
    }
  }
}

// ============= Blossom construction and expansion =============

/// Creates a new S-blossom from the cycle closed by edge `v`-`w`, whose paths
/// back through the alternating tree meet at `base`.
fn add_blossom(ctx: Ctx, s: State, base: NodeId, v: NodeId, w: NodeId) -> State {
  let bb = in_blossom(s, base)
  let id = s.next_id
  let b = Nested(id)

  let #(v_children, v_edges) =
    trace_to_base(s, in_blossom(s, v), bb, False, [], [])
  let #(w_children, w_edges) =
    trace_to_base(s, in_blossom(s, w), bb, True, [], [])
  let children = list.flatten([[bb], v_children, list.reverse(w_children)])
  let edges = list.flatten([v_edges, [#(v, w)], list.reverse(w_edges)])

  let nb =
    NestedBlossom(
      children: to_array(children),
      edges: to_array(edges),
      size: list.length(children),
      base:,
      dual: 0,
      best_edges: None,
    )
  let label_edge = case dict.get(s.label_edge, bb) {
    Ok(e) -> dict.insert(s.label_edge, b, e)
    Error(_) -> dict.delete(s.label_edge, b)
  }
  let s =
    State(
      ..s,
      next_id: id + 1,
      nested: dict.insert(s.nested, id, nb),
      parent: list.fold(children, s.parent, fn(p, c) { dict.insert(p, c, b) }),
      label: dict.insert(s.label, b, SLabel),
      label_edge:,
    )

  // T-vertices inside the new blossom become S-vertices and must be scanned.
  let s =
    list.fold(leaves(s, b), s, fn(s, x) {
      let queue = case label_of(s, in_blossom(s, x)) {
        Some(TLabel) -> [x, ..s.queue]
        _ -> s.queue
      }
      State(..s, queue:, in_blossom: dict.insert(s.in_blossom, x, b))
    })

  compute_best_edges(ctx, s, id, children)
}

/// Walks from sub-blossom `bx` back to `bb` via label edges, collecting the
/// sub-blossoms and connecting edges (in order from `bb` outwards when
/// reversed). Edges are flipped when tracing the `w` side.
fn trace_to_base(
  s: State,
  bx: Blossom,
  bb: Blossom,
  flip: Bool,
  children: List(Blossom),
  edges: List(Edge),
) -> #(List(Blossom), List(Edge)) {
  case bx == bb {
    True -> #(children, edges)
    False -> {
      let #(p, q) as e = fetch(s.label_edge, bx)
      let e = case flip {
        True -> #(q, p)
        False -> e
      }
      trace_to_base(s, in_blossom(s, p), bb, flip, [bx, ..children], [
        e,
        ..edges
      ])
    }
  }
}

fn compute_best_edges(
  ctx: Ctx,
  s: State,
  id: Int,
  children: List(Blossom),
) -> State {
  let b = Nested(id)
  let #(s, best_to) =
    list.fold(children, #(s, dict.new()), fn(acc, child) {
      let #(s, best_to) = acc
      let #(s, candidates) = case child {
        Leaf(x) -> #(s, list.map(neighbours(ctx, x), fn(y) { #(x, y) }))
        Nested(cid) -> {
          let nb = fetch(s.nested, cid)
          case nb.best_edges {
            Some(es) -> #(
              State(
                ..s,
                nested: dict.insert(
                  s.nested,
                  cid,
                  NestedBlossom(..nb, best_edges: None),
                ),
              ),
              es,
            )
            None -> #(
              s,
              list.flat_map(leaves(s, child), fn(x) {
                list.map(neighbours(ctx, x), fn(y) { #(x, y) })
              }),
            )
          }
        }
      }
      let best_to =
        list.fold(candidates, best_to, fn(best_to, k) {
          let #(i, j) = case in_blossom(s, k.1) == b {
            True -> #(k.1, k.0)
            False -> k
          }
          let bj = in_blossom(s, j)
          case bj != b && label_of(s, bj) == Some(SLabel) {
            False -> best_to
            True ->
              case dict.get(best_to, bj) {
                Ok(#(p, q)) ->
                  case slack(ctx, s, i, j) < slack(ctx, s, p, q) {
                    True -> dict.insert(best_to, bj, k)
                    False -> best_to
                  }
                Error(_) -> dict.insert(best_to, bj, k)
              }
          }
        })
      #(State(..s, best_edge: dict.delete(s.best_edge, child)), best_to)
    })

  let best_edges = dict.values(best_to)
  let best_edge =
    list.fold(best_edges, None, fn(best, e) {
      let d = slack(ctx, s, e.0, e.1)
      case best {
        Some(#(current, _)) if current <= d -> best
        _ -> Some(#(d, e))
      }
    })
  let nb = fetch(s.nested, id)
  State(
    ..s,
    nested: dict.insert(
      s.nested,
      id,
      NestedBlossom(..nb, best_edges: Some(best_edges)),
    ),
    best_edge: case best_edge {
      Some(#(_, e)) -> dict.insert(s.best_edge, b, e)
      None -> dict.delete(s.best_edge, b)
    },
  )
}

/// Dissolves top-level blossom `id` into its sub-blossoms. At the end of a
/// stage, zero-dual sub-blossoms are expanded recursively. Mid-stage, an
/// expanded T-blossom's sub-blossoms are relabelled so the alternating tree
/// stays valid.
fn expand_blossom(s: State, id: Int, end_stage: Bool) -> State {
  let nb = fetch(s.nested, id)
  let b = Nested(id)
  let s =
    dict.fold(nb.children, s, fn(s, _, child) {
      let s = State(..s, parent: dict.delete(s.parent, child))
      case child {
        Nested(cid) ->
          case end_stage && fetch(s.nested, cid).dual == 0 {
            True -> expand_blossom(s, cid, end_stage)
            False -> set_in_blossom(s, child)
          }
        Leaf(x) -> State(..s, in_blossom: dict.insert(s.in_blossom, x, child))
      }
    })
  let s = case end_stage, label_of(s, b) {
    False, Some(TLabel) -> relabel_expanded_t_blossom(s, nb, b)
    _, _ -> s
  }
  State(
    ..s,
    label: dict.delete(s.label, b),
    label_edge: dict.delete(s.label_edge, b),
    best_edge: dict.delete(s.best_edge, b),
    nested: dict.delete(s.nested, id),
  )
}

fn set_in_blossom(s: State, b: Blossom) -> State {
  State(
    ..s,
    in_blossom: list.fold(leaves(s, b), s.in_blossom, fn(acc, x) {
      dict.insert(acc, x, b)
    }),
  )
}

fn relabel_expanded_t_blossom(s: State, nb: NestedBlossom, b: Blossom) -> State {
  let #(v, w) = fetch(s.label_edge, b)
  let entry = in_blossom(s, w)
  let #(j, step) = start_and_direction(nb, entry)

  // Relabel the T/S sub-blossom pairs on the even-length path from the entry
  // child to the base.
  let #(s, j, v, w) = relabel_to_base(s, nb, j, step, v, w)

  // The base sub-blossom becomes T without propagating S to its mate (which
  // is already labelled).
  let bw = at(nb.children, nb.size, j)
  let s =
    State(
      ..s,
      label: s.label |> dict.insert(Leaf(w), TLabel) |> dict.insert(bw, TLabel),
      label_edge: s.label_edge
        |> dict.insert(Leaf(w), #(v, w))
        |> dict.insert(bw, #(v, w)),
      best_edge: dict.delete(s.best_edge, bw),
    )

  relabel_reachable(s, nb, j + step, step, entry)
}

fn relabel_to_base(
  s: State,
  nb: NestedBlossom,
  j: Int,
  step: Int,
  v: NodeId,
  w: NodeId,
) -> #(State, Int, NodeId, NodeId) {
  case j {
    0 -> #(s, j, v, w)
    _ -> {
      let #(p, q) = step_edge(nb, j, step)
      let s =
        State(
          ..s,
          label: s.label |> dict.delete(Leaf(w)) |> dict.delete(Leaf(q)),
        )
      let s = assign_label(s, w, TLabel, Some(v)) |> allow(p, q)
      let j = j + step
      let #(v, w) = step_edge(nb, j, step)
      relabel_to_base(allow(s, v, w), nb, j + step, step, v, w)
    }
  }
}

/// Continues round the blossom from the base back to the entry child, giving
/// label T to any sub-blossom that contains a vertex reached from outside.
fn relabel_reachable(
  s: State,
  nb: NestedBlossom,
  j: Int,
  step: Int,
  entry: Blossom,
) -> State {
  let bv = at(nb.children, nb.size, j)
  case bv == entry {
    True -> s
    False -> {
      let s = case label_of(s, bv) {
        Some(SLabel) -> s
        _ -> {
          let reached =
            list.find(leaves(s, bv), fn(x) { dict.has_key(s.label, Leaf(x)) })
          case reached {
            Ok(x) -> {
              let s =
                State(
                  ..s,
                  label: s.label
                    |> dict.delete(Leaf(x))
                    |> dict.delete(Leaf(fetch(s.mate, base_of(s, bv)))),
                )
              assign_label(s, x, TLabel, Some(fetch(s.label_edge, Leaf(x)).0))
            }
            Error(_) -> s
          }
        }
      }
      relabel_reachable(s, nb, j + step, step, entry)
    }
  }
}

// ============= Augmentation =============

/// Augments the matching along the path through S-vertices `v` and `w`.
fn augment_matching(s: State, v: NodeId, w: NodeId) -> State {
  s
  |> augment_from(v, w)
  |> augment_from(w, v)
}

/// Matches `v` to `partner`, then flips matched/unmatched edges back along
/// `v`'s alternating tree until reaching a single vertex.
fn augment_from(s: State, v: NodeId, partner: NodeId) -> State {
  let bs = in_blossom(s, v)
  let s = augment_if_nested(s, bs, v)
  let s = State(..s, mate: dict.insert(s.mate, v, partner))
  case dict.get(s.label_edge, bs) {
    Error(_) -> s
    Ok(#(t, _)) -> {
      let bt = in_blossom(s, t)
      let #(next, j) = fetch(s.label_edge, bt)
      let s = augment_if_nested(s, bt, j)
      let s = State(..s, mate: dict.insert(s.mate, j, next))
      augment_from(s, next, j)
    }
  }
}

fn augment_if_nested(s: State, b: Blossom, v: NodeId) -> State {
  case b {
    Nested(id) -> augment_blossom(s, id, v)
    Leaf(_) -> s
  }
}

/// Flips the alternating path inside blossom `id` from vertex `v` to the base,
/// then rotates the blossom so `v` becomes its new base.
fn augment_blossom(s: State, id: Int, v: NodeId) -> State {
  let b = Nested(id)
  let t = child_containing(s, Leaf(v), b)
  let s = augment_if_nested(s, t, v)
  let nb = fetch(s.nested, id)
  let i = index_of(nb.children, t)
  let #(j, step) = start_and_direction(nb, t)
  let s = augment_path_in_blossom(s, nb, j, step)

  let children = rotate(nb.children, nb.size, i)
  let nb =
    NestedBlossom(
      ..nb,
      children:,
      edges: rotate(nb.edges, nb.size, i),
      base: base_of(s, at(children, nb.size, 0)),
    )
  State(..s, nested: dict.insert(s.nested, id, nb))
}

fn augment_path_in_blossom(
  s: State,
  nb: NestedBlossom,
  j: Int,
  step: Int,
) -> State {
  case j {
    0 -> s
    _ -> {
      let j = j + step
      let #(w, x) = step_edge(nb, j, step)
      let s = augment_if_nested(s, at(nb.children, nb.size, j), w)
      let j = j + step
      let s = augment_if_nested(s, at(nb.children, nb.size, j), x)
      let s = State(..s, mate: s.mate |> dict.insert(w, x) |> dict.insert(x, w))
      augment_path_in_blossom(s, nb, j, step)
    }
  }
}

fn child_containing(s: State, t: Blossom, b: Blossom) -> Blossom {
  let p = fetch(s.parent, t)
  case p == b {
    True -> t
    False -> child_containing(s, p, b)
  }
}

// ============= Helpers =============

/// Going round a blossom from `child` to the base must take the even-length
/// side: forwards (wrapping) from an odd index, backwards from an even one.
/// Returns the starting index (negative when going forwards) and the step.
fn start_and_direction(nb: NestedBlossom, child: Blossom) -> #(Int, Int) {
  let i = index_of(nb.children, child)
  case int.is_odd(i) {
    True -> #(i - nb.size, 1)
    False -> #(i, -1)
  }
}

/// The edge crossed when stepping onto index `j` in direction `step`, oriented
/// in the direction of travel.
fn step_edge(nb: NestedBlossom, j: Int, step: Int) -> Edge {
  case step {
    1 -> at(nb.edges, nb.size, j)
    _ -> swap(at(nb.edges, nb.size, j - 1))
  }
}

fn swap(e: #(a, a)) -> #(a, a) {
  #(e.1, e.0)
}

fn slack(ctx: Ctx, s: State, v: NodeId, w: NodeId) -> Int {
  fetch(s.dual, v) + fetch(s.dual, w) - 2 * fetch(fetch(ctx.adj, v), w)
}

fn allow(s: State, v: NodeId, w: NodeId) -> State {
  State(..s, allowed: s.allowed |> set.insert(#(v, w)) |> set.insert(#(w, v)))
}

fn neighbours(ctx: Ctx, v: NodeId) -> List(NodeId) {
  case dict.get(ctx.adj, v) {
    Ok(targets) -> dict.keys(targets)
    Error(_) -> []
  }
}

fn in_blossom(s: State, v: NodeId) -> Blossom {
  fetch(s.in_blossom, v)
}

fn label_of(s: State, b: Blossom) -> Option(Label) {
  option.from_result(dict.get(s.label, b))
}

fn base_of(s: State, b: Blossom) -> NodeId {
  case b {
    Leaf(v) -> v
    Nested(id) -> fetch(s.nested, id).base
  }
}

fn leaves(s: State, b: Blossom) -> List(NodeId) {
  do_leaves(s, [b], [])
}

fn do_leaves(s: State, stack: List(Blossom), acc: List(NodeId)) -> List(NodeId) {
  case stack {
    [] -> acc
    [Leaf(v), ..rest] -> do_leaves(s, rest, [v, ..acc])
    [Nested(id), ..rest] ->
      do_leaves(
        s,
        dict.fold(fetch(s.nested, id).children, rest, fn(st, _, c) { [c, ..st] }),
        acc,
      )
  }
}

fn to_array(items: List(a)) -> Dict(Int, a) {
  list.index_fold(items, dict.new(), fn(acc, item, i) {
    dict.insert(acc, i, item)
  })
}

/// Circular indexing: negative and out-of-range indices wrap.
fn at(array: Dict(Int, a), size: Int, j: Int) -> a {
  let assert Ok(i) = int.modulo(j, size)
  fetch(array, i)
}

fn index_of(array: Dict(Int, a), item: a) -> Int {
  let assert Ok(i) =
    dict.fold(array, Error(Nil), fn(acc, i, x) {
      case x == item {
        True -> Ok(i)
        False -> acc
      }
    })
  i
}

/// Rotates left by `by`, so the item at index `by` moves to index 0.
fn rotate(array: Dict(Int, a), size: Int, by: Int) -> Dict(Int, a) {
  dict.fold(array, dict.new(), fn(acc, i, x) {
    let assert Ok(k) = int.modulo(i - by, size)
    dict.insert(acc, k, x)
  })
}

fn fetch(d: Dict(k, v), key: k) -> v {
  let assert Ok(value) = dict.get(d, key)
  value
}
