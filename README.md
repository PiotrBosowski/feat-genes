# feat-genes
Employing Genetic Algorithm for feature selection.


raw notes/ideas from the discord server:
the idea is to make a fully evolvable environment that will maximize feature selection, maximize training subset selection and minimize validation subset selection, possibly in a manner of aging-manner (chromosome must be old enough to be dropped out/to become a parent)

its all about performing huge amount of small trainings/validations and to cross-use the results of each in other trainings

pomiar podobieństwa chormosomów jako jeden z przerywaczy procesu

moze trzeba by zrobić taki decay starych fitnesów? średnia byłaby ważona, a waga fitnesu zmniejszałaby się z czasem

pomysł: sortowanie chromosomów od najlepszych (by najlepsze chromosomy łączyły się razem w jedno rozwiązanie)

dodać elityzm może?

symmetric-decaying-chromosome that will guarantee the symmetry between labels
batch-normalising every fitness-evaluation
uniform crossover
new chromosomes should be on average of the same size as the current population (or current best)
log avg age
check not just 1, but f.e. 9 best solutions after each epoch
Smaller ttainings
Divide chromosome and conquer
fitness = % of support vectors?
podział chromosomu na małe cegiełki
ewolucja poszczególnych cegiełek
budowa finalnego rozwiązania z najlepszych cegiełek
