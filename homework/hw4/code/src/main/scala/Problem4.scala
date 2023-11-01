package opt.heu.hw4

type Point = (Int, Int)

def manhattanDistance(p1: Point, p2: Point): Double = {
  val (x1, y1) = p1
  val (x2, y2) = p2
  Math.abs(x1 - x2) + Math.abs(y1 - y2)
}

def assignmentDistance(assignment: String): Double = {
  // identifies all edges
  val edgeStrings = Vector(
    "ab", "ag", "bc", "ch", "ci",
    "di", "de", "ef", "el", "gh",
    "gj", "hi", "il", "jk", "kl"
  )
  val locations = Vector(
    (2, 0), (2, 1), (2, 2), (2, 3),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (0, 0), (0, 1), (0, 2), (0, 3)
  ) // flat vector of locations, basically 'normal' coordinates upside down like in the handout
  
  // converts assignment to map of locations
  val locMap = assignment.zip(locations).toMap
  val edges = edgeStrings.map(s => (s(0), s(1)))
  
  // computes distances of every edge
  val distances = edges.map(edge => {
    val (p1, p2) = edge
    val p1Point = locMap(p1)
    val p2Point = locMap(p2)
    manhattanDistance(p1Point, p2Point)
  })
  
  1 / distances.sum.toFloat
}

def p4Runner(): Unit = {
  println(assignmentDistance("ceagbhdkfjli"))
  println(assignmentDistance("cligjakbfdeh"))
}
