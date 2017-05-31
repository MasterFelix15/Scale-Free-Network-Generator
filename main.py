import numpy
from numpy import linalg as la
import math
import random
import matplotlib.pyplot as plt


# analyze a given graph, return the degree map and distribution histogram
def analyzeGraph(graph):
    N = 0
    nodes = []
    for i in range(graph.shape[0]):
        degreeCount = 0
        for j in range(graph.shape[1]):
            if graph[i, j] == 1:
                degreeCount += 1
                N += 1
        nodes.append(degreeCount)
    distribution = nodes.copy()
    for i in range(len(nodes)):
        distribution[i] /= N
    return nodes, distribution

# plot a given graph, in terms of its degree distribution
def plotThisGraph(graph):
    graph = numpy.array(graph)
    nodes, histogram = analyzeGraph(graph)
    plt.plot(histogram, 'ro')
    plt.ylim(0, 0.1)
    plt.show()

# find all node indexes of a given degree from a given list of nodes
def findPointsOfDegree(nodes, degree):
    pointList = []
    for i in range(len(nodes)):
        if nodes[i] == degree:
            pointList.append(i)
    return pointList

# generate a BA graph
def BAGenerator(n0, m, N):
    degree = numpy.zeros((N, 1))
    mark = numpy.zeros((N, 1))
    num = numpy.zeros((N, 1))
    A = numpy.zeros((N, N))
    for i in range(n0):
        degree[i] = n0-1
        for j in range(n0):
            if i != j:
                A[i][j] = 1
                A[j][i] = 1
    for i in range(n0, N):
        degree[i] = m
        for j in range(i-1):
            mark[j]=0
        for j in range(i):
            num[j]=0
        sum = 0
        for j in range(i-1):
            sum += degree[j]
            num[j+1] = sum
        count = 0
        while count < m:
            for j in range(i-1):
                rd = random.random()
                f = rd * sum
                if num[j] < f < num[j+1] and mark[j] == 0:
                    mark[j] = 1
            count = count + 1
        for j in range(i-1):
            if mark[j] == 1:
                degree[j] = degree[j] + 1
                A[i][j] = 1
                A[j][i] = 1
    return A


def RandomGraphGenerator(N):
    graph = numpy.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if random.random() < 0.5:
                graph[i, j] = 1
    for i in range(N):
        for j in range(i):
            graph[i, j] = graph[j, i]
        graph[i, i] = 0
    return graph


def removeDuplicates(degreeList):
    listNoDuplicates = []
    for i in range(len(degreeList)):
        if not(degreeList[i] in listNoDuplicates):
            listNoDuplicates.append(degreeList[i])
    return listNoDuplicates

def kevinAlgo(graph, gamma, cycle):
    addCount = 0
    delCount = 0
    movCount = 0
    QCount = 1
    zeta = 0
    for index in range(1, 10000):
        zeta += math.pow(index, -1 * gamma)
    A = numpy.matrix(graph, dtype=int)
    Qpast = 2
    invalidResultCount = 0
    loopCount = 0
    while True:
        if invalidResultCount > 100:
            break
        if loopCount > cycle:
            break
        invalidResultCount += 1
        loopCount += 1
        foundValidChange = False
        validChangeType = -1
        nodes, distribution = analyzeGraph(A)
        d = min(nodes)
        N = A.shape[0]
        # generate CP
        CP = numpy.eye(N, dtype=int)
        for power in range(2, N):
            CP += la.matrix_power(A, power)

        k = random.choice(nodes)
        NDnodes = removeDuplicates(nodes)
        for x in range(len(NDnodes)):
            if foundValidChange:
                break
            l = NDnodes[x]
            klist = findPointsOfDegree(nodes, k)
            kp1list = findPointsOfDegree(nodes, k + 1)
            km1list = findPointsOfDegree(nodes, k - 1)
            llist = findPointsOfDegree(nodes, l)
            lp1list = findPointsOfDegree(nodes, l + 1)
            lm1list = findPointsOfDegree(nodes, l - 1)

            nk = len(klist)
            nkp1 = len(kp1list)
            nkm1 = len(km1list)
            nl = len(llist)
            nlp1 = len(lp1list)
            nlm1 = len(lm1list)

            pk = math.pow(k - d + 1, -1 * gamma) / zeta
            pkp1 = math.pow(k - d + 2, -1 * gamma) / zeta
            try:
                pkm1 = math.pow(k - d, -1 * gamma) / zeta
            except ValueError:
                pkm1 = 0
            pl = math.pow(l - d + 1, -1 * gamma) / zeta
            plp1 = math.pow(l - d + 2, -1 * gamma) / zeta
            try:
                plm1 = math.pow(l - d, -1 * gamma) / zeta
            except ValueError:
                plm1 = 0
            i = random.choice(klist)
            for y in range(len(llist)):
                if foundValidChange:
                    break
                j = llist[y]
                if A[i, j] == 0:  # not connected, add edge
                    validChangeType = 0
                    if abs(k - l) > 1:
                        Q = nk * nl * pkp1 * plp1 / (pk * pl * (nkp1 + 1) * (nlp1 + 1))
                    elif k + 1 == l:
                        Q = nk * plp1 / (pk * (nlp1 + 1))
                    elif k == l + 1:
                        Q = nl * pkp1 / (pl * (nkp1 + 1))
                    elif k == l and nk >= 2:
                        Q = nk * (nk - 1) * math.pow(pkp1, 2) / (math.pow(pk, 2) * (nkp1 + 2) * (nkp1 + 1))
                elif A[i, j] == 1 and CP[i, j] > 0:  # connected, remove edge
                    validChangeType = 1
                    if abs(k - l) > 1:
                        Q = nk * nl * pkm1 * plm1 / (pk * pl * (nkm1 + 1) * (nlm1 + 1))
                    elif k - 1 == l:
                        Q = nk * plm1 / (pk * (nlm1 + 1))
                    elif k == l - 1:
                        Q = nl * pkm1 / (pl * (nkm1 + 1))
                    elif k == l and nk >= 2:
                        Q = nk * (nk - 1) * math.pow(pkm1, 2) / math.pow(pk, 2) * (nkm1 + 2) * (nkm1 + 1)
                else:
                    for m in range(1, nk):
                        validChangeType = m
                        aiT = A[i].transpose()
                        aj = A[j]
                        leftSide = numpy.dot(aiT, aj)[0, 0] - A[i, j]
                        rightSide = nk - m
                        if leftSide <= rightSide:
                            nkpm = len(findPointsOfDegree(nodes, k + m))
                            nkmm = len(findPointsOfDegree(nodes, k - m))
                            nlpm = len(findPointsOfDegree(nodes, l + m))

                            pkpm = math.pow(k - d + m + 1, -1 * gamma) / zeta
                            try:
                                pkmm = math.pow(k - d - m + 1, -1 * gamma) / zeta
                            except ValueError:
                                pkmm = 0
                            plpm = math.pow(l - d + m + 1, -1 * gamma) / zeta
                            if k - l != 0 and k - l != m and k - l != 2 * m:
                                Q = nk * pkmm * nl * plpm / (pk * (nkmm + 1) * pl * (nlpm + 1))
                            elif k == l and nk >= 2:
                                Q = nk * (nk - 1) * pkmm * pkpm / (math.pow(pk, 2) * (nkmm + 1) * (nkpm + 1))
                            elif k == l + m:
                                Q = 1
                            elif k - m == l + m:
                                Q = nk * nl * math.pow(pkmm, 2) / (pk * pl * (nkmm + 1) * (nkmm + 2))
                if Q and Q > 0 and 0 < math.log(Q) < Qpast:
                    print("a valid first change:" + str([i, j]) + str(Qpast) + "::" + str(validChangeType))
                    QCount += 1
                    Qpast = (Qpast * (QCount-1) + math.log(Q)) / QCount
                    foundValidChange = True
                    invalidResultCount = 0
                    if validChangeType == 0:
                        A[i, j] = 1
                        A[j, i] = 1
                        addCount += 1
                    elif validChangeType == 1:
                        A[i, j] = 0
                        A[j, i] = 0
                        delCount += 1
                    elif validChangeType > 1:  # rewire m connections with node i to node j
                        random.shuffle(klist)
                        for index in range(validChangeType):
                            g = klist[index]
                            A[g, i] = 0
                            A[i, g] = 0
                            A[j, g] = 1
                            A[g, j] = 1
                            movCount += 1
    plotThisGraph(A)
    print("Gamma Value: " + str(gamma))
    print("Add Operation Count: " + str(addCount))
    print("Delete Operation Count: " + str(delCount))
    print("Gift Operation Count: " + str(movCount))
    print("Total Operation Count: " + str(addCount + delCount + movCount))


def runThisProgram(n0, m, N, gamma, cycle):
    graph = BAGenerator(n0, m, N)
    plotThisGraph(graph)
    kevinAlgo(graph, gamma, cycle)


runThisProgram(4, 3, 50, 2.5, 1000)


