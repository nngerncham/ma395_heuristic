package opt.heu.hw4

import scala.annotation.tailrec
import scala.util.Random


def partiallyMappedCrossover(parent1: String, parent2: String): String = {
  val cutPoint = Random.between(1, parent1.length)
  val (p1Left, p1Right) = parent1.splitAt(cutPoint)
  val (p2Left, p2Right) = parent2.splitAt(cutPoint)

  @tailrec
  def scanLeftParent(leftRemaining: String, currentChar: Char, acc: String): String = {
    if leftRemaining.isEmpty then
      if !p2Right.contains(currentChar) then acc + currentChar
      else scanLeftParent("", p1Right(p2Right.indexOf(currentChar)), acc)
    else if !p2Right.contains(currentChar) then scanLeftParent(leftRemaining.tail, leftRemaining.head, acc + currentChar)
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
    if found.contains(currentIdx) then acc
    else getOneCycle(parent2.indexOf(parent1(currentIdx)), found + currentIdx, acc :+ currentIdx)
  }

  val (_, cycles: Vector[Vector[Int]]) = (0 until parent1.length).foldLeft((Set(), Vector[Vector[Int]]()))((acc, i) => {
    val (found, cycles) = acc
    if found.contains(i) then acc // if already found, skip
    else { // otherwise, identify cycle starting at i
      val cycle = getOneCycle(i, Set(), Vector())
      (found ++ cycle, cycles :+ cycle)
    }
  })

  // puts together the final string
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
