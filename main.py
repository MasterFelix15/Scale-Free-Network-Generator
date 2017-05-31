import numpy
from numpy import linalg as la
import math
import random
import matplotlib.pyplot as plt
import plotter


# analyze a given graph, return the degree map and distribution histogram
def analyze_graph(graph):
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
def plot_this_graph(graph):
    graph = numpy.array(graph)
    nodes, histogram = analyze_graph(graph)
    plt.plot(histogram, 'ro')
    plt.show()


# find all node indexes of a given degree from a given list of nodes
def find_points_of_degree(nodes, degree):
    pointList = []
    for i in range(len(nodes)):
        if nodes[i] == degree:
            pointList.append(i)
    return pointList


# generate a BA graph
def ba_generator(n0, m, N):
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


def random_graph_generator(N):
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


def remove_duplicates(degreeList):
    listNoDuplicates = []
    for i in range(len(degreeList)):
        if not(degreeList[i] in listNoDuplicates):
            listNoDuplicates.append(degreeList[i])
    return listNoDuplicates


def get_relink_candidates(node_i, node_j, graph):
    relink_candidates = []
    for index in range(graph.shape[0]):
        if graph[node_i, index] == 1 and graph[node_j, index] == 0:
            relink_candidates.append(index)
    return relink_candidates


def generate_true_scale_free_network(graph, gamma, cycle):

    # monitoring variables initialization
    cycle_count = 0
    invalid_cycle_count = 0
    add_count = 0
    del_count = 0
    mov_count = 0

    # step 1: Get graph
    graph = numpy.matrix(graph, dtype=int)
    number_of_nodes = graph.shape[0]

    while True:
        cycle_count += 1
        if cycle_count > cycle or invalid_cycle_count > 50:
            break

        # step 1.1: Calculate CP for this graph
        cp = numpy.eye(number_of_nodes, dtype=int)
        for power in range(2, number_of_nodes):
            cp += la.matrix_power(graph, power)

        # step 2: Compute the node-degree histogram n(G)
        ng, pg = analyze_graph(graph)

        # step 2.1 Compute the degree-node histogram n with n(G)
        n = []
        for index in range(2 * number_of_nodes):
            n.append(0)
        for index in range(len(ng)):
            n[ng[index]] += 1

        # step 2.2 Compute p for all nodes
        zeta = 0
        for index in range(1, 10000):
            zeta += math.pow(index, -1 * gamma)
        p = []
        for index in range(2 * number_of_nodes):
            try:
                p.append(math.pow(index - min(ng) + 1, -1 * gamma) / zeta)
            except ValueError:
                p.append(0)

        # step 3: Choose a degree k such that n[k] > 0
        k = random.choice(ng)
        k_list = find_points_of_degree(ng, k)

        # step 4: Find best Q from all possible changes made with this given k with all l and m
        best_q = 0
        best_solution = 'undefined'
        no_duplicate_ng = remove_duplicates(ng)
        for l_index in range(len(no_duplicate_ng)):
            l = no_duplicate_ng[l_index]
            l_list = find_points_of_degree(ng, l)
            for i_index in range(len(k_list)):
                i = k_list[i_index]
                for j_index in range(len(l_list)):
                    j = l_list[j_index]
                    q = 0
                    m = -2
                    if graph[i, j] == 0:  # not connected, add edge
                        m = -1
                        if abs(k - l) > 1:
                            q = n[k] * n[l] * p[k + 1] * p[l + 1] / (p[k] * p[l] * (n[k + 1] + 1) * (n[l + 1] + 1))
                        elif k + 1 == l:
                            q = n[k] * p[l + 1] / (p[k] * (n[l + 1] + 1))
                        elif k == l + 1:
                            q = n[l] * p[k + 1] / (p[l] * (n[k + 1] + 1))
                        elif k == l and n[k] >= 2:
                            q = n[k] * (n[k] - 1) * math.pow(p[k + 1], 2) / (
                            math.pow(p[k], 2) * (n[k + 1] + 2) * (n[k + 1] + 1))
                    elif graph[i, j] == 1 and cp[i, j] > 0:  # connected, remove edge
                        m = 0
                        if abs(k - l) > 1:
                            q = n[k] * n[l] * p[k - 1] * p[l - 1] / (p[k] * p[l] * (n[k - 1] + 1) * (n[l - 1] + 1))
                        elif k - 1 == l:
                            q = n[k] * p[l - 1] / (p[k] * (n[l - 1] + 1))
                        elif k == l - 1:
                            q = n[l] * p[k - 1] / (p[l] * (n[k - 1] + 1))
                        elif k == l and n[k] >= 2:
                            q = n[k] * (n[k] - 1) * math.pow(p[k - 1], 2) / math.pow(p[k], 2) * (n[k - 1] + 2) * (n[k - 1] + 1)
                    else:
                        for m in range(1, n[k]):
                            ai_t = graph[i].transpose()
                            aj = graph[j]
                            left_side = numpy.dot(ai_t, aj)[0, 0] - graph[i, j]
                            right_side = n[k] - m
                            if left_side <= right_side:
                                if k - l != 0 and k - l != m and k - l != 2 * m:
                                    q = n[k] * p[k - m] * n[l] * p[l + m] / (
                                    p[k] * (n[k - m] + 1) * p[l] * (n[l + m] + 1))
                                elif k == l and n[k] >= 2:
                                    q = n[k] * (n[k] - 1) * p[k - m] * p[k + m] / (
                                    math.pow(p[k], 2) * (n[k - m] + 1) * (n[k + m] + 1))
                                elif k == l + m:
                                    q = 1
                                elif k - m == l + m:
                                    q = n[k] * n[l] * math.pow(p[k - m], 2) / (
                                    p[k] * p[l] * (n[k - m] + 1) * (n[k - m] + 2))
                    if q > 0 and math.log(q) > best_q:
                        best_q = math.log(q)
                        print("best log(q) value has been updated to " + str(math.log(q)))
                        best_solution = [i, j, m]

        # step 5: Make the valid change to G and return to step 2
        if best_solution != 'undefined':
            i, j, m = best_solution
            if m == -1:
                print("connect node " + str(i) + " to node " + str(j))
                add_count += 1
                graph[i, j] = 1
                graph[j, i] = 1
            elif m == 0:
                print("disconnect node " + str(i) + " to node " + str(j))
                del_count += 1
                graph[i, j] = 0
                graph[j, i] = 0
            elif m >= 1:
                relink_candidates = get_relink_candidates(i, j, graph)
                random.shuffle(relink_candidates)
                gift_count = 0
                while gift_count < m:
                    try:
                        g = relink_candidates[gift_count]
                        if cp[i, g] == 0:  # test whether removing node g from node i will result in disconnected graphs
                            print("gifting node " + str(g) + " from node " + str(i) +
                                  "will result in disconnected graph, abort")
                        else:
                            print("gift node " + str(g) + " from node " + str(i))
                            mov_count += 1
                            graph[g, i] = 0
                            graph[i, g] = 0
                            graph[j, g] = 1
                            graph[g, j] = 1
                            gift_count += 1
                    except IndexError:
                        print("fail to gift nodes")
                        break
            invalid_cycle_count = 0
        else:
            print('fail to find a valid change in this cycle with k as: ' + str(k))
            invalid_cycle_count += 1
    print("Total Add Operations: " + str(add_count))
    print("Total Remove Operations: " + str(del_count))
    print("Total gift Operations: " + str(mov_count))
    return graph


def run_this_program(n0, m, N, gamma, cycle):
    graph = ba_generator(n0, m, N)
    plot_this_graph(graph)
    plotter.draw_graph(graph)
    graph = generate_true_scale_free_network(graph, gamma, cycle)
    plot_this_graph(graph)
    plotter.draw_graph(graph)


run_this_program(5, 5, 100, 2.5, 50)
