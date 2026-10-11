import gleam/int
import gleam/list
import gleam/set
import gleeunit
import pbt/qcheck_generators
import qcheck
import yog/matching
import yog/model.{type Graph}

pub fn main() {
  gleeunit.main()
}

/// Dense graphs with few distinct weights, so that odd cycles, blossoms and
/// ties are common.
fn dense_graph_generator(min_weight: Int) {
  use num_nodes <- qcheck.bind(qcheck.bounded_int(1, 10))
  use num_edges <- qcheck.bind(qcheck.bounded_int(0, 25))
  use edges <- qcheck.map(qcheck.fixed_length_list_from(
    qcheck_generators.edge_triple_generator_custom(
      num_nodes,
      qcheck.bounded_int(min_weight, 12),
    ),
    num_edges,
  ))
  let graph =
    int.range(
      from: 0,
      to: num_nodes,
      with: model.new(model.Undirected),
      run: fn(g, i) { model.add_node(g, i, Nil) },
    )
  list.fold(edges, graph, fn(g, e) {
    let assert Ok(g) = model.add_edge(g, from: e.0, to: e.1, with: e.2)
    g
  })
}

fn edge_list(graph: Graph(n, Int)) -> List(#(Int, Int, Int)) {
  model.all_edges(graph)
  |> list.filter(fn(e) { e.0 < e.1 })
}

/// Best #(cardinality, weight) over all matchings, comparing by `better`.
fn brute_force(
  edges: List(#(Int, Int, Int)),
  used: set.Set(Int),
  better: fn(#(Int, Int), #(Int, Int)) -> Bool,
) -> #(Int, Int) {
  case edges {
    [] -> #(0, 0)
    [#(u, v, w), ..rest] -> {
      let skip = brute_force(rest, used, better)
      case set.contains(used, u) || set.contains(used, v) {
        True -> skip
        False -> {
          let #(c, total) =
            brute_force(rest, used |> set.insert(u) |> set.insert(v), better)
          let take = #(c + 1, total + w)
          case better(take, skip) {
            True -> take
            False -> skip
          }
        }
      }
    }
  }
}

fn by_weight(a: #(Int, Int), b: #(Int, Int)) -> Bool {
  a.1 > b.1
}

fn by_cardinality_then_weight(a: #(Int, Int), b: #(Int, Int)) -> Bool {
  a.0 > b.0 || { a.0 == b.0 && a.1 > b.1 }
}

/// Asserts `pairs` is a valid matching in `graph` and returns its
/// #(cardinality, weight).
fn score(graph: Graph(n, Int), pairs: List(#(Int, Int))) -> #(Int, Int) {
  let vertices = list.flat_map(pairs, fn(p) { [p.0, p.1] })
  assert set.size(set.from_list(vertices)) == list.length(vertices)
  let total =
    list.fold(pairs, 0, fn(acc, p) {
      assert p.0 < p.1
      let assert Ok(w) = model.edge_data(graph, p.0, p.1)
      acc + w
    })
  #(list.length(pairs), total)
}

pub fn maximum_weight_matching_is_optimal_test() {
  use graph <- qcheck.given(dense_graph_generator(1))
  let assert Ok(pairs) = matching.maximum_weight_matching(graph, fn(w) { w })
  let #(_, weight) = score(graph, pairs)
  assert weight == brute_force(edge_list(graph), set.new(), by_weight).1
}

pub fn maximum_weight_matching_skips_negative_edges_test() {
  use graph <- qcheck.given(dense_graph_generator(-12))
  let assert Ok(pairs) = matching.maximum_weight_matching(graph, fn(w) { w })
  let #(_, weight) = score(graph, pairs)
  assert weight == brute_force(edge_list(graph), set.new(), by_weight).1
}

pub fn max_cardinality_weight_matching_is_optimal_test() {
  use graph <- qcheck.given(dense_graph_generator(-12))
  let assert Ok(pairs) =
    matching.max_cardinality_weight_matching(graph, fn(w) { w })
  assert score(graph, pairs)
    == brute_force(edge_list(graph), set.new(), by_cardinality_then_weight)
}

pub fn maximum_matching_is_maximum_cardinality_test() {
  use graph <- qcheck.given(dense_graph_generator(1))
  let assert Ok(pairs) = matching.maximum_matching(graph)
  let #(cardinality, _) = score(graph, pairs)
  assert cardinality
    == brute_force(edge_list(graph), set.new(), by_cardinality_then_weight).0
}
