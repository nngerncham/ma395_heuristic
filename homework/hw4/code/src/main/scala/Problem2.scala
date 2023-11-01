package opt.heu.hw4

import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Random

import java.io.File
import com.github.tototoshi.csv._

val REPS = 500

def mutateP2(probMutate: Double)(chromosome: String): String = {
  chromosome.map(bit => {
    val flip = Map('0' -> '1', '1' -> '0')
    if math.random < probMutate then flip(bit) else bit
  })
}

def fitnessP2(x: String): Double = {
  val xInt = Integer.parseInt(x, 2)
  2 * math.pow(xInt, 3) - 240 * math.pow(xInt, 2) + 7200 * xInt + 2000
}

def cutcatenate(parent1: String, parent2: String): String = {
  val cutPoint = Random.nextInt(parent1.length - 1) + 1
  val (p1Left, _) = parent1.splitAt(cutPoint)
  val (_, p2Right) = parent2.splitAt(cutPoint)
  p1Left + p2Right
}

def intToBinaryString(x: Int): String = {
  val binary = x.toBinaryString
  val padding = "0" * (6 - binary.length)
  padding + binary
}

def p2Runner(): Unit = {
  val results10Future = (0 until REPS).map(_ => Future {
    rouletteWheel(
      initialPopulation = (0 until 6)
        .map(_ => Random.nextInt(64))
        .map(intToBinaryString)
        .toVector, // initial population
      populationSize = 6,
      numGeneration = 10,
      numOffspring = 6,
      fitnessFunction = fitnessP2,
      crossover = cutcatenate,
      mutate = mutateP2(0.03),
    )
  })
  val results10 = Await.result(Future.sequence(results10Future), scala.concurrent.duration.Duration.Inf)

  // part 1
  val avgFitness = results10
    .map(_.avgPopFitness)
    .transpose
    .map(_.sum / REPS.toDouble)
    .map(_.round)
  print("Average fitness: ")
  avgFitness.foreach(e => print(s"& $e "))
  println()

  // part 2
  val bestFitness = results10
    .map(_.bestPopFitness)
    .transpose
    .map(_.sum / REPS.toDouble)
    .map(_.round)
  print("Best fitness: ")
  bestFitness.foreach(e => print(s"& $e "))
  println()

  // part 3
  val finalPopulations10 = results10.map(_.finalPopulation)
  val mean010 = finalPopulations10.map(_.count(_.startsWith("010"))).sum / REPS.toDouble
  val mean11 = finalPopulations10.map(_.count(_.startsWith("11"))).sum / REPS.toDouble
  println(s"Mean of schema [010***]: $mean010")
  println(s"Mean of schema [11****]: $mean11")

  // part 4
  val results250 = Await.result(Future.sequence((0 until REPS).map(_ => Future {
    rouletteWheel(
      initialPopulation = (0 until 6)
        .map(_ => Random.nextInt(64))
        .map(intToBinaryString)
        .toVector, // initial population
      populationSize = 6,
      numGeneration = 1000,
      numOffspring = 6,
      fitnessFunction = fitnessP2,
      crossover = cutcatenate,
      mutate = mutateP2(0.03),
    )
  })), scala.concurrent.duration.Duration.Inf)
  val fileResult = new File("/Users/nawat/muic/ma395_heuristic/homework/hw4/results/p2/part4.csv")
  val writer = CSVWriter.open(fileResult)
  writer.writeRow(List("Generation", "Average fitness", "Best fitness"))
  results250.foreach(result => {
    val data = result.avgPopFitness
      .zip(result.bestPopFitness)
      .zipWithIndex
      .map { case ((avg, best), i) => Seq(i, avg, best) }
    writer.writeAll(data)
  })

  // part 5
  val initialPopulation = (0 until 6)
    .map(_ => Random.nextInt(64))
    .map(intToBinaryString)
    .toVector

  Vector(0.03, 0.05, 0.075, 0.1).foreach(prob => {
    val trialResult = Await.result(Future.sequence((0 until REPS).map(_ => Future {
      rouletteWheel(
        initialPopulation = initialPopulation,
        populationSize = 6,
        numGeneration = 250,
        numOffspring = 6,
        fitnessFunction = fitnessP2,
        crossover = cutcatenate,
        mutate = mutateP2(prob),
      )
    })), scala.concurrent.duration.Duration.Inf)

    val fileResult = new File(s"/Users/nawat/muic/ma395_heuristic/homework/hw4/results/p2/part5pr$prob.csv")
    val writer = CSVWriter.open(fileResult)
    writer.writeRow(List("Generation", "Average fitness", "Best fitness"))
    trialResult.foreach(result => {
      val data = result.avgPopFitness
        .zip(result.bestPopFitness)
        .zipWithIndex
        .map { case ((avg, best), i) => Seq(i, avg, best) }
      writer.writeAll(data)
    })
  })

  // part 6
  Vector(3, 6, 10).foreach(numChild => {
    val bigInitialPopulation = (0 until numChild)
      .map(_ => Random.nextInt(64))
      .map(intToBinaryString)
      .toVector
    val trialResult = Await.result(Future.sequence((0 until REPS).map(_ => Future {
      rouletteWheel(
        initialPopulation = bigInitialPopulation,
        populationSize = numChild,
        numGeneration = 250,
        numOffspring = numChild,
        fitnessFunction = fitnessP2,
        crossover = cutcatenate,
        mutate = mutateP2(0.075),
      )
    })), scala.concurrent.duration.Duration.Inf)

    val fileResult = new File(s"/Users/nawat/muic/ma395_heuristic/homework/hw4/results/p2/part5pop$numChild.csv")
    val writer = CSVWriter.open(fileResult)
    writer.writeRow(List("Generation", "Average fitness", "Best fitness"))
    trialResult.foreach(result => {
      val data = result.avgPopFitness
        .zip(result.bestPopFitness)
        .zipWithIndex
        .map { case ((avg, best), i) => Seq(i, avg, best) }
      writer.writeAll(data)
    })
  })
}
