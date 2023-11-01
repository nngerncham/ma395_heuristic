package opt.heu.hw4

import scala.util.Random

def tournamentGA(initialPopulation: Vector[String],
                 populationSize: Int,
                 numGeneration: Int,
                 numOffspring: Int,
                 fitnessFunction: String => Double,
                 crossover: (String, String) => String,
                 mutate: String => String,
                 probCrossover: Double = 1.0,
                ): GAResults = {
  assert(initialPopulation.length == populationSize)

  // Accumulator is population, avgPopFitness, bestPopFitness
  val fitnessPopulation = initialPopulation.map(fitnessFunction)
  val (finalPopulation, avgPopFitness, bestPopFitness) =
    (0 until numGeneration).foldLeft((
      initialPopulation,
      Vector(fitnessPopulation.sum / populationSize.toDouble),
      Vector(fitnessPopulation.max),
    ))((acc, _) => {
      // unpacks accumulator
      val (currentPop, avgPopFitness, bestPopFitness) = acc

      // computes cumulative fitness and apply crossovers
      val crossovers = (0 until numOffspring).map(_ => {
        val (p1Candidates, p2Candidates) = Random.shuffle(currentPop)
          .map(e => (e, fitnessFunction(e)))
          .splitAt(populationSize / 2)
        val parent1 = p1Candidates.maxBy(_._2)._1
        val parent2 = p2Candidates.maxBy(_._2)._1
        if math.random < probCrossover then crossover(parent1, parent2) else parent1
      })

      // mutates the crossovers to generate offsprings
      val offsprings = crossovers.map(child => mutate(child))

      // generates new population
      val newPopulation = (currentPop ++ offsprings).sortBy(fitnessFunction).reverse.take(populationSize)
      val newPopFitness = newPopulation.map(fitnessFunction)
      (
        newPopulation,
        avgPopFitness :+ newPopFitness.sum / populationSize.toDouble,
        bestPopFitness :+ newPopulation.map(fitnessFunction).max,
      )
    })

  GAResults(finalPopulation, avgPopFitness, bestPopFitness)
}
