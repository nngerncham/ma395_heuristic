package opt.heu.hw4

import scala.annotation.tailrec
import scala.util.Random


def partiallyMappedCrossover(parent1: String, parent2: String): String = {
  // randomizes and splits the parents
  val cutPoint = Random.between(1, parent1.length)
  val (p1Left, p1Right) = parent1.splitAt(cutPoint)
  val (p2Left, p2Right) = parent2.splitAt(cutPoint)

  @tailrec
  def scanLeftParent(leftRemaining: String, currentChar: Char, acc: String): String = {
    // if current char not in right substring of P2, add it to the accumulator and return
    // otherwise, recursively find the corresponding gene in left substring of P1
    if leftRemaining.isEmpty then
      if !p2Right.contains(currentChar) then acc + currentChar
      else scanLeftParent("", p1Right(p2Right.indexOf(currentChar)), acc)
    // if current char not in right substring of P2, add it to the accumulator and move on to next gene
    else if !p2Right.contains(currentChar) then scanLeftParent(leftRemaining.tail, leftRemaining.head, acc + currentChar)
    // otherwise, recursively find the corresponding gene in left substring of P1
    else scanLeftParent(leftRemaining, p1Right(p2Right.indexOf(currentChar)), acc)
  }

  scanLeftParent(p1Left.tail, p1Left.head, "") + p2Right
}

def orderCrossover(parent1: String, parent2: String): String = {
  val cutPoint = Random.between(1, parent1.length)
  val (p1Left, p1Right) = parent1.splitAt(cutPoint)
  p1Left + parent2.filterNot(p1Left.contains)
}

def cycleCrossover(parent1: String, parent2: String): String = {
  @tailrec
  def getOneCycle(currentIdx: Int, found: Set[Int], acc: Vector[Int]): Vector[Int] = {
    // if we've already found this index, we've completed the cycle
    if found.contains(currentIdx) then acc
    // otherwise, add this index to the cycle and continue
    else getOneCycle(parent2.indexOf(parent1(currentIdx)), found + currentIdx, acc :+ currentIdx)
  }

  val (_, cycles: Vector[Vector[Int]]) = (0 until parent1.length).foldLeft((Set(), Vector[Vector[Int]]()))((acc, i) => {
    // unpacks the accumulator
    val (found, cycles) = acc

    // if i already has a cycle, skip
    if found.contains(i) then acc
    // otherwise, identify cycle starting at i
    else {
      val cycle = getOneCycle(i, Set(), Vector())
      (found ++ cycle, cycles :+ cycle)
    }
  })

  // flips between the two parents to put the corresponding genes in to the final string
  var finalString = " ".repeat(parent1.length)
  var parentIdx = 0
  val flip = Map(1 -> 0, 0 -> 1)
  val parents = Vector(parent1, parent2)
  for (i <- cycles.indices) {
    val parentToUse = parents(parentIdx)
    val cycle = cycles(i)
    cycle.foreach(idx => finalString = finalString.updated(idx, parentToUse(idx)))
    parentIdx = flip(parentIdx)
  }

  finalString
}
