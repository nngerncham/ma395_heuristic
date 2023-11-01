package opt.heu.hw4

import scala.util.Random
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global

import java.io.File
import com.github.tototoshi.csv._

val REPS = 1000
val numGens = 1000

def swap(probMutate: Double)(string: String): String = {
  val (i, j) = (Random.nextInt(string.length), Random.nextInt(string.length))
  if Random.nextDouble() < probMutate then string.updated(i, string(j)).updated(j, string(i))
  else string
}

def p5Runner(): Unit = {
  Vector(partiallyMappedCrossover, orderCrossover, cycleCrossover).zip(Vector("pmx", "order", "cycle"))
    .foreach((crossover, name) => {
      val initialPopulation = (0 until 10).map(_ => Random.shuffle("abcdefghijkl").toString).toVector
      val rouletteTrials = Await.result(Future.sequence((0 until REPS).map(_ => Future {
        rouletteWheelGA(
          initialPopulation = initialPopulation,
          populationSize = 10,
          numGeneration = numGens,
          numOffspring = 10,
          fitnessFunction = assignmentDistance,
          crossover = crossover,
          mutate = swap(0.05),
          probCrossover = 0.9,
        )
      })), scala.concurrent.duration.Duration.Inf)
      val fileResult1 = new File(s"/Users/nawat/muic/ma395_heuristic/homework/hw4/results/p56/roulette$name.csv")
      val writer1 = CSVWriter.open(fileResult1)
      writer1.writeRow(List("Generation", "Average fitness", "Best fitness"))
      rouletteTrials.foreach(result => {
        val data = result.avgPopFitness
          .zip(result.bestPopFitness)
          .zipWithIndex
          .map { case ((avg, best), i) => Seq(i, avg, best) }
        writer1.writeAll(data)
      })

      val tournamentTrials = Await.result(Future.sequence((0 until REPS).map(_ => Future {
        tournamentGA(
          initialPopulation = initialPopulation,
          populationSize = 10,
          numGeneration = numGens,
          numOffspring = 10,
          fitnessFunction = assignmentDistance,
          crossover = partiallyMappedCrossover,
          mutate = swap(0.05),
          probCrossover = 0.9,
        )
      })), scala.concurrent.duration.Duration.Inf)
      val fileResult2 = new File(s"/Users/nawat/muic/ma395_heuristic/homework/hw4/results/p56/tournament$name.csv")
      val writer2 = CSVWriter.open(fileResult2)
      writer2.writeRow(List("Generation", "Average fitness", "Best fitness"))
      tournamentTrials.foreach(result => {
        val data = result.avgPopFitness
          .zip(result.bestPopFitness)
          .zipWithIndex
          .map { case ((avg, best), i) => Seq(i, avg, best) }
        writer2.writeAll(data)
      })
    })
}
