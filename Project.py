import numpy as np
import networkx as nx
import random as rd
import collections as cl
import matplotlib.pyplot as plt
import time
import xlrd
import config
import pandas as pd
from copy import copy, deepcopy
from fractions import Fraction


EXPERIMENTS_counter = 0
AVERAGE_operators = []
IMPROVED = []


def read_ways():
    file = xlrd.open_workbook('./Лабиринт_Архив.xlsx')

    sheet = file.sheet_by_index(0)

    ways = []
    rows = sheet.nrows

    for row in range(rows):
        if sheet.row(row)[4].value == 1.0:
            ways.append(sheet.row(row)[5].value)
    return ways


def generate_way(d, Mat):
    c = list(range(26))
    f = [22]
    gen_way = "Ы"
    while f[0] != 25:
        f = rd.choices(c, Mat[f[0]], k = 1)
        gen_way += d[f[0]]
    return gen_way


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bfs(graph, root):
    paths = [0] * 26
    paths[root] = -1
    visited, queue = set(), cl.deque([root])
    visited.add(root)
    while queue: 
        vertex = queue.popleft()
        for i in range(len(graph[vertex])): 
            if graph[vertex][i] != 0 and i not in visited: 
                visited.add(i) 
                queue.append(i)
                paths[i] = vertex
    return paths


def Lev_distance(a, b):
    n = len(a)
    m = len(b)
    if n > m:
        (a, b) = (b, a)
        (n, m) = (m, n)
    row = range(n + 1)
    for i in range(1, m + 1):
        prev_row = row
        row = [i] + [0] * n
        for j in range(1, n + 1):
            k = prev_row[j] + 1
            g = row[j - 1] + 1
            f = prev_row[j - 1]
            if a[j - 1] != b[i - 1]:
                f += 1
            row[j] = min(k, g, f)
    return row[n]


def Find_Lev_dist(array, s):
    ans = -1
    for i in array:
        k = Lev_distance(i, s)
        if ans == -1 or k < ans:
            ans = k
    return ans



def inverse(a, i, d, pairs, set_right, set_left, Mat):
    k = a[:i + 1].rfind("Ы")
    while k != 0:
        if a[k - 1] == "Т" or a[k - 1] == "Р":
            break
        else:
            k = a[:k].rfind("Ы")
    s = a[k:i]
    m = a[i + 1:].find("Ы")
    if m != -1:
        return a[:i + 1] + s[::-1] + a[m + 1:]
    return a[:i + 1] + s[::-1] + generate_way(d, Mat)[1:-1]


def compress(a, i, d, pairs, set_right, set_left, Mat):
    j = 0
    counter_a = 0
    counter_g = 0
    while j < i:
        c = j + 1
        if a[j] == "А":
            counter_a += 1;
        if a[j] == "Г":
            counter_g += 1;
        while c < i:
            if (c > j + 1 and Mat[d.index(a[j])][d.index(a[c])] == 1):
                a = a[:j + 1] + a[c:]
                i -= c - j - 1
                j -= 1
                break
            if a[c] == "А":
                if counter_a > 0:
                    if (c > j + 1 and Mat[d.index(a[j])][d.index(a[c])] == 1):
                        a = a[:j + 1] + a[c:]
                        i -= c - j - 1
                        j -= 1
                        break
                else:
                    break
            if a[c] == "Г":
                if counter_g > 0:
                    if (c > j + 1 and Mat[d.index(a[j])][d.index(a[c])] == 1):
                        a = a[:j + 1] + a[c:]
                        i -= c - j - 1
                        j -= 1
                        break
                else:
                    break
            elif a[c] == a[j]:
                a = a[:j] + a[c:]
                i -= (c - j)
                j -= 1
                break
            c += 1
        j += 1
    return a


def symmetry(a, i, d, pairs, set_right, set_left, Mat):
    k = i
    s = a[:k + 1]
    if a[i] == "T":
        while a[i] != "Ы":
            if a[i] in set_left:
                break
            i -= 1
        if a[i] == "Ы":
            while i <= k:
                s += pairs[a[i]]
                i += 1
    if a[i] == "Р":
        while a[i] != "Ы":
            if a[i] in set_right:
                break
            i -= 1
        if a[i] == "Ы":
            while i <= k:
                s += pairs[a[i]]
                i += 1
    return s + a[k + 1:]


def circle(a, i, d, pairs, set_right, set_left, Mat):
    new_Mat = deepcopy(Mat)
    s = ''
    a_pos = a[i:].find("А")
    g_pos = a[i:].find("Г")
    d_pos = a[i:].find("Д")
    k_pos = a[i:].find("К")
    h_pos = a[i:].find("Х")
    ya_pos = a[i:].find("Я")
    ts_pos = a[i:].find("Ц")
    z_pos = a[i:].find("З")
    if a_pos != -1:
        a_pos += i
        new_Mat[d.index("Г")][d.index("Ц")] = 0
        new_Mat[d.index("Ц")][d.index("Х")] = 0
        new_Mat[d.index("Х")][d.index("А")] = 0
        paths = bfs(new_Mat, 3)
        j = 0
        while paths[j] != -1:
            s += d[j]
            j = paths[j]
        s += "ГЦХ"
        return a[:a_pos + 1] + s[::-1] + a[a_pos + 1:]
    elif g_pos != -1:
        g_pos += i
        new_Mat[d.index("А")][d.index("Х")] = 0
        new_Mat[d.index("Х")][d.index("Ц")] = 0
        new_Mat[d.index("Ц")][d.index("Г")] = 0
        paths = bfs(new_Mat, 0)
        j = 3
        while paths[j] != -1:
            s += d[j]
            j = paths[j]
        s += "АХЦ"
        return a[:g_pos + 1] + s[::-1] + a[g_pos + 1:]
    elif d_pos != -1 or k_pos != -1 or h_pos != -1:
        a_pos = max(d_pos, k_pos, h_pos) + i
        a = a[:a_pos + 1] + "А" + a[a_pos + 1:]
        a_pos += 1
        new_Mat[d.index("Г")][d.index("Ц")] = 0
        new_Mat[d.index("Ц")][d.index("Х")] = 0
        new_Mat[d.index("Х")][d.index("А")] = 0
        paths = bfs(new_Mat, 3)
        j = 0
        while paths[j] != -1:
            s += d[j]
            j = paths[j]
        s += "ГЦХ"
        return a[:a_pos + 1] + s[:0:-1] + a[a_pos - 1] + a[a_pos + 1:]
    elif ya_pos != -1 or ts_pos != -1 or z_pos != -1:
        g_pos = max(ya_pos, ts_pos, z_pos) + i
        a = a[:g_pos + 1] + "Г" + a[g_pos + 1:]
        g_pos += 1
        new_Mat[d.index("А")][d.index("Х")] = 0
        new_Mat[d.index("Х")][d.index("Ц")] = 0
        new_Mat[d.index("Ц")][d.index("Г")] = 0
        paths = bfs(new_Mat, 0)
        j = 3
        while paths[j] != -1:
            s += d[j]
            j = paths[j]
        s += "АХЦ"
        return a[:g_pos + 1] + s[:0:-1] + a[g_pos - 1] + a[g_pos + 1:]
    return a


def choose_operator(ulta_lucky, counter_operators, counter_improved_operators, counter, best_lev, optimal, operators, probability_of_operators, a, i, d, pairs, set_right, set_left, Mat):
    index_operators = list(range(4))
    f = rd.choices(index_operators, probability_of_operators, k=1)
    string = ultra_lucky
    #posl.append(a)
    new_path = operators[f[0]](a, i, d, pairs, set_right, set_left, Mat)
    lev = Find_Lev_dist(optimal, new_path)
    counter_operators[operators.index(operators[f[0]])] += 1
    #distances.append(lev)
    #op_posl.append(operators[f[0]].__name__)
    if best_lev > lev:
        best_lev = lev
        #probability_of_operators[f[0]] = sigmoid((best_lev - lev) / len(a))
        counter += 1
        counter_improved_operators[operators.index(operators[f[0]])] += 1
        string += str(f[0])
    else:
        #probability_of_operators[f[0]] = sigmoid((lev - best_lev) / len(a))
        new_path = a
    #best_posl.append(new_path)
    probability_of_operators[f[0]] = sigmoid((best_lev - lev) / len(a))
    return new_path, counter, string




pairs = dict([("Р", "T"), ("Т", "Р"), ("А", "Г"), ("Г", "А"), ("Х", "Ц"), ("Ц", "Х"), ("Д", "З"), ("З", "Д"), ("К", "Я"), ("Я", "К"), 
         ("С", "У"), ("У", "С"), ("М", "И"), ("И", "М"), ("Л", "Э"), ("Э", "Л"), ("Щ", "Ш"), ("Ш", "Щ"), ("Ж", "Е"), ("Е", "Ж"), 
         ("Б", "В"), ("В", "Б"), ("Ф", "Ф"), ("О", "О"), ("Ы", "Ы")])
set_right = {"Т", "Ц", "Г", "З", "У", "Я", "М", "Щ", "Э", "В", "Ж"}
set_left = {"Р", "Б", "Е", "Ш", "Л", "И", "С", "К", "Д", "А", "Х"}
d = ["А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "К", "Л", "М", "О", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ш", "Щ", "Ы", "Э", "Я", "."]
graph = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],    
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
optimal_way = ["ЫОИКАХЦГЗУЩТ", "ЫОМЯГЦХАДСШР", "ЫОИКАХЦГЯЭЖТ", "ЫОМЯГЦХАКЛЕР", "ЫОИКДАХЦГЗУЩТ", "ЫОМЯЗГЦХАДСШР"]
operators = [inverse, compress, symmetry, circle]
ways = read_ways()
counter_lucky = 0
lucky_ways = []
ultra_lucky = ""
#s = "ЫОФЦГЦХАКИМЭЖТЩУЗГЦФМЯГЦХАКЛЕР"
#posl = [s]
#best_posl = [s]
#op_posl = [None]
#distances = []
#distances.append(Find_Lev_dist(optimal_way, s))
#probability_of_operators = [1.0, 1.0, 1.0, 1.0]
#counter = 0
#all_counter = 0
#counter_operators = [0, 0, 0, 0]
#counter_improved_operators = [0, 0, 0, 0]
#EXPERIMENTS_counter += 1
#f = open('gg.txt', 'w')
#while counter < 6 and Find_Lev_dist(optimal_way, s) > 3 and all_counter < 20:
 #   all_counter += 1
  #  i = rd.choices(list(range(len(s))), [1] * len(s), k=1)
   # s, counter, ultra_lucky = choose_operator(ultra_lucky, counter_operators, counter_improved_operators, counter, Find_Lev_dist(optimal_way, s), optimal_way, operators, probability_of_operators, s, i[0], d, pairs, set_right, set_left, graph)
    #print("Current best way: {}".format(s))
    #print("Levenshteins distance: {}".format(Find_Lev_dist(optimal_way, s)))   
    #print(probability_of_operators, counter)
    #print(counter_improved_operators)
    #print(counter_operators)
#df = pd.DataFrame({'Последовательности к которым применяем оператор': posl,
 #                  'Оператор':  op_posl,
#                   'Последовательности после модификации': best_posl,
 #                  'Расстояние Левенштейна': distances})
#f.write(df.to_string())
for c in range(10):
    s = "Ы" + ways[c]
 #   s = "ЫОФХАКИМЯГЦХАДСШРЕЛИМЯГЯЭЖТЩУЗГЦФМЯГЦХАДСШЕЛИФЦГЦХАДСШРШСДХЦГЯЭЖТ"
    probability_of_operators = [1.0, 1.0, 1.0, 1.0]
    counter = 0
    all_counter = 0
    counter_operators = [0, 0, 0, 0]
    counter_improved_operators = [0, 0, 0, 0]
    EXPERIMENTS_counter += 1
    while counter < 6 and Find_Lev_dist(optimal_way, s) > 3 and all_counter < 20:
        all_counter += 1
        i = rd.choices(list(range(len(s))), [1] * len(s), k=1)
        s, counter, ultra_lucky = choose_operator(ultra_lucky, counter_operators, counter_improved_operators, counter, Find_Lev_dist(optimal_way, s), optimal_way, operators, probability_of_operators, s, i[0], d, pairs, set_right, set_left, graph)
        #print("Current best way: {}".format(s))
        #print("Levenshteins distance: {}".format(Find_Lev_dist(optimal_way, s)))   
        #print(probability_of_operators, counter)
        #print(counter_improved_operators)
        #print(counter_operators)
        #time.sleep(1)
    if Find_Lev_dist(optimal_way, s) <= 3 or counter >= 6:
        counter_lucky += 1
        lucky_ways.append(ultra_lucky)
    AVERAGE_operators.append(counter_operators)
    if sum(counter_improved_operators) != 0:
        IMPROVED.append(counter_improved_operators)
ans1 = [0] * 4
ans2 = [0] * 4
st = len(AVERAGE_operators)
for i in range(len(AVERAGE_operators)):
    if sum(AVERAGE_operators[i]) == 0:
        st -= 1
    for j in range(4):
        if sum(AVERAGE_operators[i]) != 0:
            ans1[j] += Fraction(AVERAGE_operators[i][j], sum(AVERAGE_operators[i]))
for i in range(4):
    if st != 0:
        ans1[i] /= st
for i in range(len(IMPROVED)):
    for j in range(4):
        if sum(IMPROVED[i]) != 0:
            ans2[j] += Fraction(IMPROVED[i][j], sum(IMPROVED[i]))
for i in range(4):
    ans2[i] /= len(IMPROVED)
#print(ans1)
#print(sum(ans1))
#print(ans2)
#print(sum(ans2))
fig, ax = plt.subplots()
ax.bar(range(1, 5), ans2)
ax.set_facecolor('seashell')
fig.set_facecolor('floralwhite')
fig.set_figwidth(3)
fig.set_figheight(10)
plt.show()
#print(counter_lucky)
#print(lucky_ways)