import pygame
from pygame.locals import *

import numpy
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import random


def get_node_degrees(graph):
    N = 0
    nodes = []
    for i in range(graph.shape[0]):
        degreeCount = 0
        for j in range(graph.shape[1]):
            if graph[i, j] == 1:
                degreeCount += 1
                N += 1
        nodes.append(degreeCount)
    return nodes


def generate_random_positions(number_of_nodes):
    params = []
    for i in range(number_of_nodes):
        u = random.random()
        v = random.random()
        theta = 2 * math.pi * u
        phi = math.acos(2 * v - 1)
        x = math.cos(theta) * math.sin(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(phi)
        params.append(numpy.matrix([x, y, z]))
    return params


def compute_coordinates(graph):
    number_of_nodes = graph.shape[0]
    params = generate_random_positions(number_of_nodes)
    position = []
    position_updated = []
    seperated_head = 0
    node_degrees = get_node_degrees(graph)
    for i in range(number_of_nodes):
        position.append(numpy.matrix([0, 0, 0]))
        position_updated.append(False)
    for i in range(number_of_nodes):
        if not position_updated[i]:
            position[i] = params[i] * seperated_head
            seperated_head += 0.2
            position_updated[i] = True
        for j in range(number_of_nodes):
            if graph[i, j] == 1 and not position_updated[j]:
                position[j] = position[i] + params[j] * node_degrees[i]/max(node_degrees) * 1.5
                position_updated[j] = True
    return position


def draw(position, graph):
    node_size = 0.02
    for i in range(len(position)):
        glPushMatrix()
        glTranslatef(position[i][0, 0], position[i][0, 1], position[i][0, 2])
        glutSolidSphere(node_size, 20, 20)
        glPopMatrix()
    for i in range(graph.shape[0]):
        for j in range(i+1, graph.shape[0]):
            if graph[i, j] == 1:
                glBegin(GL_LINES)
                glVertex3fv((position[i][0, 0], position[i][0, 1], position[i][0, 2]))
                glVertex3fv((position[j][0, 0], position[j][0, 1], position[j][0, 2]))
                glEnd()


def visualize_graph(graph):
    pygame.init()
    display = (600, 600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    graph = numpy.matrix(graph)
    position = compute_coordinates(graph)

    while True:
        glRotatef(0.5, 3, 1, 1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        draw(position, graph)
        pygame.display.flip()
        pygame.time.wait(10)


def read_graph(name):
    f = open(name + '.graph', 'r')
    graph = []
    for line in f:
        line = list(numpy.array(line.split(' ')).astype(int))
        graph.append(line)
    return numpy.matrix(graph)


#graph = read_graph('initial')
graph = read_graph('optimized')
visualize_graph(graph)