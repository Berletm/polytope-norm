import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from typing import List, Tuple

import os

REPORT_PTH = R"../report"

# var2
VERTICES = [
    (5, 3, 0),      # v1
    (3, 0, 6),      # v2
    (0, 2, 7),      # v3 
    (33/5, 0, 0),   # v4
    (0, 171/32, 0), # v5
    (0, 0, 107/13), # v6 
]

QUADRANTS = [
    (1, 1, 1),    # 1
    (-1, 1, 1),   # 2
    (1, -1, 1),   # 3
    (1, 1, -1),   # 4
    (-1, -1, 1),  # 5
    (-1, 1, -1),  # 6
    (1, -1, -1),  # 7
    (-1, -1, -1)  # 8
]

ALL_QUADRANTS_INDICES = list(range(8))

FACETS = [
    (1, 2, 3),
    (1, 2, 4), 
    (1, 3, 5),
    (2, 3, 6)
]

POINTS = [
    (7, -6, 9), # a
    (-6, 6, 7)  # b
]

class Vec3:
    def __init__(self, vec: Tuple[int|float]):
        self.x = vec[0]
        self.y = vec[1]
        self.z = vec[2]
    
    def to_list(self) -> List:
        return [self.x, self.y, self.z]
        
class Facet:
    def __init__(self, points: Tuple[Vec3]):
        self.points = points
    
class Quadrant:
    def __init__(self, facets: List[Facet]):
        self.facets = facets

class W:
    def __init__(self, initial_vertices:List = VERTICES):
        self.base_vertices: List = initial_vertices
        
        self.facets:    List[Facet]    = []
        self.quadrants: List[Quadrant] = []
        self.vertices:  List[Vec3]     = []
         
        self.__create_W()
    
        
    def __create_W(self) -> None:
        for signs in QUADRANTS:
            s1, s2, s3 = signs
            
            vertices = [Vec3((s1 * v[0], s2 * v[1], s3 * v[2])) for v in self.base_vertices]
            
            facets   = [Facet((vertices[f[0] - 1], vertices[f[1] - 1], vertices[f[2] - 1])) for f in FACETS]
            
            self.facets.extend(facets)
            self.quadrants.append(Quadrant(facets))
            self.vertices.extend(vertices)
    
    def draw_W(self, quadrants_indices=ALL_QUADRANTS_INDICES) -> None:
        fig = plt.figure(figsize=(24, 8))
        ax  = plt.axes(projection="3d")
        
        self.__draw_axis(ax)
                    
        for i, q in enumerate(self.quadrants):
            if i in quadrants_indices:
                for f in q.facets:
                    verts = [p.to_list() for p in f.points]
                    poly = Poly3DCollection([verts], alpha=0.7)
                    poly.set_edgecolor("dimgray")
                    poly.set_facecolor("lightgray")
                    ax.add_collection3d(poly)
            
                    verts = np.array(verts)
        
        points = np.array(POINTS)
        ax.scatter(points[:, 0], points[:, 1], c="black", s=25)
        
        ax.set_aspect('equal')
        ax.axis("off")
        plt.show()
        
    def __draw_axis(self, ax: plt.Axes) -> None:
        origin = [0, 0, 0]
        axis_length = 20
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']
        
        for i in range(3):
            direction = [0, 0, 0]
            direction[i] = axis_length
            neg_direction = [-d for d in direction]
            ax.quiver(*origin, *neg_direction, color=colors[i], linewidth=2, arrow_length_ratio=0.0, alpha=0.8)
            
            ax.quiver(*origin, *direction, color=colors[i], linewidth=2, arrow_length_ratio=0.1, alpha=0.8)
            
            label_pos = [0, 0, 0]
            label_pos[i] = axis_length * 1.1
            ax.text(*label_pos, labels[i], color=colors[i], fontsize=24, fontweight='bold')
    
    def dump_tex_tables(self, pth: str) -> None:
        pth2vertices  = os.path.join(pth, "vertices.tex")
        pth2allvertices  = os.path.join(pth, "all_vertices.tex")
        pth2quadrants = os.path.join(pth, "quadrants.tex")
        
        self.__dump_base_vertices(pth2vertices)
        self.__dump_all_vertices(pth2allvertices)
        self.__dump_unique_vertices("1231")
        
    def __dump_base_vertices(self, pth: str) -> None:
        with open(pth, 'w+', encoding='utf-8') as f:
            f.write(r'\begin{table}[h]' + '\n')
            f.write(r'    \centering' + '\n')
            
            f.write(r'    \caption{Вершины первого квадранта}' + '\n')
            f.write(r'    \begin{tabular}{c||ccc}' + '\n')
            f.write(r'        \toprule' + '\n')
            
            f.write(r'        \textbf{№} & \textbf{$x$} & \textbf{$y$} & \textbf{$z$} \\'+ '\n')
            f.write(r'        \midrule' + '\n')
            
            for i, row in enumerate(self.base_vertices, 1):
                x, y, z = row
                x = r"$\frac{33}{5}$" if abs(float(x) - float(33/5)) < 1e-9 else x
                y = r"$\frac{171}{32}$" if abs(float(y) - float(171/32)) < 1e-9 else y
                z = r"$\frac{107}{13}$" if abs(float(z) - float(107/13)) < 1e-9 else z
                line = f"        {i} & {x} & {y} & {z} \\\\\n"
                f.write(line)
            
            f.write(r'        \bottomrule' + '\n')
            f.write(r'    \end{tabular}' + '\n')
            f.write(r'\end{table}' + '\n')
        
    def __dump_all_vertices(self, pth: str) -> None:
        lets = ["a", "b", "c"]
        with open(pth, 'w+', encoding='utf-8') as f:
            f.write(r'\begin{table}[h]' + '\n')
            f.write(r'    \centering' + '\n')
            l = 0
            r = 16
            for i in range(1, 4):
                f.write(r'    \begin{subtable}{0.3\textwidth}' + '\n')
                f.write(r'        \centering' + '\n')
                f.write(r'        \begin{tabular}{c||ccc}' + '\n')
                f.write(r'            \toprule' + '\n')
                f.write(r'            \textbf{№} & \textbf{$x$} & \textbf{$y$} & \textbf{$z$} \\'+ '\n')
                f.write(r'            \midrule' + '\n') 
                
                for j in range(l, r, 1):
                    v = self.vertices[j]
                    
                    x = r"$\frac{33}{5}$" if (abs(float(v.x) - float(33/5)) < 1e-9 or abs(float(-v.x) - float(33/5)) < 1e-9) else v.x
                    y = r"$\frac{171}{32}$" if (abs(float(v.y) - float(171/32)) < 1e-9 or abs(float(-v.y) - float(171/32)) < 1e-9) else v.y
                    z = r"$\frac{107}{13}$" if (abs(float(v.z) - float(107/13)) < 1e-9 or abs(float(-v.z) - float(107/13)) < 1e-9) else v.z
                    
                    
                    f.write(f"            {j} & {x} & {y} & {z} \\\\\n")
                
                f.write(r"        \end{tabular}" + "\n")
                f.write(r'            \caption{Вершины группы №' + f"{i}" + r"}" + "\n")
                f.write(r"            \label{tab:sub_" + f"{lets[i-1]}" "}" + "\n")
                f.write(r"    \end{subtable}" + "\n")
                if j != 3:
                    f.write(r"    \hfill" + "\n")

                l += 16
                r += 16
            f.write(r"    \caption{Все вершины многогранника}" + "\n")
            f.write(r"    \label{tab:main}" + "\n")
            f.write(r"\end{table}" + "\n")
    
    def __dump_unique_vertices(self, pth: str) -> None:
        pass
    
    
def main() -> None:
    cover = W()
    
    cover.draw_W([1])
    cover.dump_tex_tables(REPORT_PTH)


if __name__ == "__main__":
    main()