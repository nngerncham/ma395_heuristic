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
                  crossover: (String, String) => String,
                  fitnessFunction: String => Double,
                  mutate: (String, Double) => String,
                  probCrossover: Double = 1.0,
                  probMutation: Double = 0.03,
                 ): GAResults = {
  var population = initialPopulation
  val fitnessPopulation = population.map(fitnessFunction)
  val cumulativeFitness = fitnessPopulation.scanLeft(0.0)(_ + _).tail
  for (i <- 0 until numGeneration) {
    val crossovers = (0 until numOffspring).map(_ => {
      val parent1 = population(rouletteWheelSelection(cumulativeFitness))
      val parent2 = population(rouletteWheelSelection(cumulativeFitness))
      if math.random < probCrossover then crossover(parent1, parent2) else parent1
    })
    val fitnessCrossover = crossovers.map(fitnessFunction)
    val mutated = crossovers.map(child => {
      val mutated = mutate(child, probMutation)
      fitnessFunction(mutated)
    })
    population = (population ++ crossovers)
      .sortWith((a, b) => fitnessFunction(a) > fitnessFunction(b))
      .take(populationSize)
  }
  GAResults(population.head, fitnessFunction(population.head))
}
