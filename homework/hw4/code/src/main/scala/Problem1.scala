package opt.heu.hw4

import breeze.plot._

def p1Runner(): Unit = {
  val cumulativeFitness = Vector(math.pow(25.0, 2), math.pow(44, 2), math.pow(53, 2), math.pow(56, 2)).scan(0.0)(_ + _).tail
  val trialResults = (0 until 100).map(_ => rouletteWheelSelection(cumulativeFitness) + 1)
  val figure = Figure()
  val plot = figure.subplot(0)
  plot += hist(trialResults)
  plot.xlabel = "Individual"
  plot.ylabel = "Frequency"
  figure.saveas("/Users/nawat/muic/ma395_heuristic/homework/hw4/images/p1/roulette_histogram.png")
}
