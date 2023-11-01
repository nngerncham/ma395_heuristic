package opt.heu.hw4

def rouletteWheelSelection(cumulativeFitness: Vector[Double]): Int = {
  val totalFitness = cumulativeFitness.last
  val randomValue = math.random * totalFitness
  cumulativeFitness.indexWhere(_ >= randomValue)
}

def rouletteWheel(initialPopulation: Vector[String],
                  populationSize: Int,
                  numGeneration: Int,
                  numOffspring: Int,
                  fitnessFunction: String => Double,
                  crossover: (String, String) => String,
                  mutate: String => String,
                  probCrossover: Double = 1.0,
                 ): GAResults = {
  assert(initialPopulation.length == populationSize)

  // Accumulator is population, avgPopFitness, bestPopFitness, populations
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
      val cumulativeFitness = currentPop.map(fitnessFunction).scanLeft(0.0)(_ + _).tail
      val crossovers = (0 until numOffspring).map(_ => {
        val parent1 = currentPop(rouletteWheelSelection(cumulativeFitness))
        val parent2 = currentPop(rouletteWheelSelection(cumulativeFitness))
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
