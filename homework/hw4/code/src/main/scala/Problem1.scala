package opt.heu.hw4

import breeze.plot._

def p1Runner(): Unit = {
  val cumulativeFitness = Vector(math.pow(25.0, 2), math.pow(44, 2), math.pow(53, 2), math.pow(56, 2)).scan(0.0)(_ + _).tail
  val trialResults = (0 until 100).map(_ => rouletteWheelSelection(cumulativeFitness) + 1)
  val trialResults2 = (0 until 100_000).map(_ => rouletteWheelSelection(cumulativeFitness) + 1)

  val figure = Figure()
  val plot1 = figure.subplot(0)
  plot1 += hist(trialResults)
  plot1.xlabel = "Individual"
  plot1.ylabel = "Frequency"

  val figure2 = Figure()
  val plot2 = figure2.subplot(0)
  plot2 += hist(trialResults2)
  plot2.xlabel = "Individual"
  plot2.ylabel = "Frequency"

  figure.saveas("/Users/nawat/muic/ma395_heuristic/homework/hw4/images/p1/roulette_histogram.png")
  figure2.saveas("/Users/nawat/muic/ma395_heuristic/homework/hw4/images/p1/roulette_histogram2.png")
}
