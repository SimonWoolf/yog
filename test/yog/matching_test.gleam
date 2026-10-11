import gleam/list
import gleeunit/should
import yog
import yog/matching
import yog/model

fn weighted(edges: List(#(Int, Int, Int))) -> model.Graph(Nil, Int) {
  let assert Ok(graph) =
    list.fold(edges, yog.undirected(), fn(g, e) {
      g |> yog.add_node(e.0, Nil) |> yog.add_node(e.1, Nil)
    })
    |> yog.add_edges(edges)
  graph
}

fn max_weight(edges: List(#(Int, Int, Int))) -> List(#(Int, Int)) {
  let assert Ok(m) =
    matching.maximum_weight_matching(weighted(edges), fn(w) { w })
  m
}

fn max_card_weight(edges: List(#(Int, Int, Int))) -> List(#(Int, Int)) {
  let assert Ok(m) =
    matching.max_cardinality_weight_matching(weighted(edges), fn(w) { w })
  m
}

// ============= Basic cases =============

pub fn empty_graph_test() {
  matching.maximum_weight_matching(yog.undirected(), fn(w) { w })
  |> should.equal(Ok([]))
}

pub fn directed_graph_is_error_test() {
  let assert Ok(graph) =
    yog.directed()
    |> yog.add_node(1, Nil)
    |> yog.add_node(2, Nil)
    |> yog.add_edges([#(1, 2, 1)])

  matching.maximum_matching(graph)
  |> should.be_error()
}

pub fn isolated_nodes_test() {
  let graph =
    yog.undirected()
    |> yog.add_node(1, Nil)
    |> yog.add_node(2, Nil)

  matching.maximum_matching(graph)
  |> should.equal(Ok([]))
}

pub fn self_loop_ignored_test() {
  max_weight([#(0, 0, 100)])
  |> should.equal([])
}

pub fn single_edge_test() {
  max_weight([#(0, 1, 1)])
  |> should.equal([#(0, 1)])
}

pub fn two_path_test() {
  max_weight([#(1, 2, 10), #(2, 3, 11)])
  |> should.equal([#(2, 3)])
}

pub fn path_prefers_heavy_middle_edge_test() {
  max_weight([#(1, 2, 5), #(2, 3, 11), #(3, 4, 5)])
  |> should.equal([#(2, 3)])
}

pub fn square_test() {
  max_weight([#(1, 4, 2), #(2, 3, 2), #(1, 2, 1), #(3, 4, 4)])
  |> should.equal([#(1, 2), #(3, 4)])
}

pub fn negative_weights_test() {
  let edges = [#(1, 2, 2), #(1, 3, -2), #(2, 3, 1), #(2, 4, -1), #(3, 4, -6)]

  max_weight(edges)
  |> should.equal([#(1, 2)])

  max_card_weight(edges)
  |> should.equal([#(1, 3), #(2, 4)])
}

pub fn min_weight_perfect_matching_via_negation_test() {
  max_card_weight([#(1, 2, -1000), #(2, 3, -2), #(3, 4, -3000)])
  |> should.equal([#(1, 2), #(3, 4)])
}

// ============= Cardinality =============

pub fn odd_cycle_cardinality_test() {
  // A 5-cycle with a pendant vertex on each cycle vertex has a perfect
  // matching that augmenting-path search without blossoms can miss.
  let assert Ok(graph) =
    yog.undirected()
    |> yog.add_node(1, Nil)
    |> yog.add_node(2, Nil)
    |> yog.add_node(3, Nil)
    |> yog.add_node(4, Nil)
    |> yog.add_node(5, Nil)
    |> yog.add_node(6, Nil)
    |> yog.add_edges([
      #(1, 2, Nil),
      #(2, 3, Nil),
      #(3, 4, Nil),
      #(4, 5, Nil),
      #(5, 1, Nil),
      #(1, 6, Nil),
    ])

  let assert Ok(m) = matching.maximum_matching(graph)
  list.length(m)
  |> should.equal(3)
}

pub fn petersen_has_perfect_matching_test() {
  let outer = [#(0, 1), #(1, 2), #(2, 3), #(3, 4), #(4, 0)]
  let spokes = [#(0, 5), #(1, 6), #(2, 7), #(3, 8), #(4, 9)]
  let inner = [#(5, 7), #(7, 9), #(9, 6), #(6, 8), #(8, 5)]
  let edges =
    list.flatten([outer, spokes, inner])
    |> list.map(fn(e) { #(e.0, e.1, 1) })

  let assert Ok(m) = matching.maximum_matching(weighted(edges))
  list.length(m)
  |> should.equal(5)
}

// ============= Blossom cases =============
// These come from Van Rantwijk's mwmatching test suite (also used by
// NetworkX); each is constructed to exercise a particular blossom code path.

pub fn s_blossom_test() {
  max_weight([#(1, 2, 8), #(1, 3, 9), #(2, 3, 10), #(3, 4, 7)])
  |> should.equal([#(1, 2), #(3, 4)])

  max_weight([
    #(1, 2, 8),
    #(1, 3, 9),
    #(2, 3, 10),
    #(3, 4, 7),
    #(1, 6, 5),
    #(4, 5, 6),
  ])
  |> should.equal([#(1, 6), #(2, 3), #(4, 5)])
}

pub fn s_t_blossom_test() {
  max_weight([
    #(1, 2, 9),
    #(1, 3, 8),
    #(2, 3, 10),
    #(1, 4, 5),
    #(4, 5, 4),
    #(1, 6, 3),
  ])
  |> should.equal([#(1, 6), #(2, 3), #(4, 5)])

  max_weight([
    #(1, 2, 9),
    #(1, 3, 8),
    #(2, 3, 10),
    #(1, 4, 5),
    #(4, 5, 3),
    #(1, 6, 4),
  ])
  |> should.equal([#(1, 6), #(2, 3), #(4, 5)])

  max_weight([
    #(1, 2, 9),
    #(1, 3, 8),
    #(2, 3, 10),
    #(1, 4, 5),
    #(4, 5, 3),
    #(3, 6, 4),
  ])
  |> should.equal([#(1, 2), #(3, 6), #(4, 5)])
}

pub fn nested_s_blossom_test() {
  max_weight([
    #(1, 2, 9),
    #(1, 3, 9),
    #(2, 3, 10),
    #(2, 4, 8),
    #(3, 5, 8),
    #(4, 5, 10),
    #(5, 6, 6),
  ])
  |> should.equal([#(1, 3), #(2, 4), #(5, 6)])
}

pub fn nested_s_blossom_relabel_test() {
  max_weight([
    #(1, 2, 10),
    #(1, 7, 10),
    #(2, 3, 12),
    #(3, 4, 20),
    #(3, 5, 20),
    #(4, 5, 25),
    #(5, 6, 10),
    #(6, 7, 10),
    #(7, 8, 8),
  ])
  |> should.equal([#(1, 2), #(3, 4), #(5, 6), #(7, 8)])
}

pub fn nested_s_blossom_expand_test() {
  max_weight([
    #(1, 2, 8),
    #(1, 3, 8),
    #(2, 3, 10),
    #(2, 4, 12),
    #(3, 5, 12),
    #(4, 5, 14),
    #(4, 6, 12),
    #(5, 7, 12),
    #(6, 7, 14),
    #(7, 8, 12),
  ])
  |> should.equal([#(1, 2), #(3, 5), #(4, 6), #(7, 8)])
}

pub fn s_blossom_relabel_expand_test() {
  max_weight([
    #(1, 2, 23),
    #(1, 5, 22),
    #(1, 6, 15),
    #(2, 3, 25),
    #(3, 4, 22),
    #(4, 5, 25),
    #(4, 8, 14),
    #(5, 7, 13),
  ])
  |> should.equal([#(1, 6), #(2, 3), #(4, 8), #(5, 7)])
}

pub fn nested_s_blossom_relabel_expand_test() {
  max_weight([
    #(1, 2, 19),
    #(1, 3, 20),
    #(1, 8, 8),
    #(2, 3, 25),
    #(2, 4, 18),
    #(3, 5, 18),
    #(4, 5, 13),
    #(4, 7, 7),
    #(5, 6, 7),
  ])
  |> should.equal([#(1, 8), #(2, 3), #(4, 7), #(5, 6)])
}

pub fn nasty_blossom_relabel_multiple_ways_test() {
  max_weight([
    #(1, 2, 45),
    #(1, 5, 45),
    #(2, 3, 50),
    #(3, 4, 45),
    #(4, 5, 50),
    #(1, 6, 30),
    #(3, 9, 35),
    #(4, 8, 35),
    #(5, 7, 26),
    #(9, 10, 5),
  ])
  |> should.equal([#(1, 6), #(2, 3), #(4, 8), #(5, 7), #(9, 10)])
}

pub fn nasty_blossom_relabel_multiple_ways_variant_test() {
  max_weight([
    #(1, 2, 45),
    #(1, 5, 45),
    #(2, 3, 50),
    #(3, 4, 45),
    #(4, 5, 50),
    #(1, 6, 30),
    #(3, 9, 35),
    #(4, 8, 26),
    #(5, 7, 40),
    #(9, 10, 5),
  ])
  |> should.equal([#(1, 6), #(2, 3), #(4, 8), #(5, 7), #(9, 10)])
}

pub fn nasty_blossom_least_slack_test() {
  max_weight([
    #(1, 2, 45),
    #(1, 5, 45),
    #(2, 3, 50),
    #(3, 4, 45),
    #(4, 5, 50),
    #(1, 6, 30),
    #(3, 9, 35),
    #(4, 8, 28),
    #(5, 7, 26),
    #(9, 10, 5),
  ])
  |> should.equal([#(1, 6), #(2, 3), #(4, 8), #(5, 7), #(9, 10)])
}

pub fn nasty_nested_blossom_augmenting_test() {
  max_weight([
    #(1, 2, 45),
    #(1, 7, 45),
    #(2, 3, 50),
    #(3, 4, 45),
    #(4, 5, 95),
    #(4, 6, 94),
    #(5, 6, 94),
    #(6, 7, 50),
    #(1, 8, 30),
    #(3, 11, 35),
    #(5, 9, 36),
    #(7, 10, 26),
    #(11, 12, 5),
  ])
  |> should.equal([#(1, 8), #(2, 3), #(4, 6), #(5, 9), #(7, 10), #(11, 12)])
}

pub fn nasty_blossom_expand_recursively_test() {
  max_weight([
    #(1, 2, 40),
    #(1, 3, 40),
    #(2, 3, 60),
    #(2, 4, 55),
    #(3, 5, 55),
    #(4, 5, 50),
    #(1, 8, 15),
    #(5, 7, 30),
    #(7, 6, 10),
    #(8, 10, 10),
    #(4, 9, 30),
  ])
  |> should.equal([#(1, 2), #(3, 5), #(4, 9), #(6, 7), #(8, 10)])
}
