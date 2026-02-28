import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from typing import List, Tuple

import os

REPORT_PTH = R"../report"
IMAGES_PTH = R"../report/images"

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
    (-6, 6, 7),  # b
]

CENTROID = np.array([0, 0, 0], dtype=np.float32)

class Vec3:
    def __init__(self, vec: Tuple[int|float]):
        self.x = vec[0]
        self.y = vec[1]
        self.z = vec[2]
    
    def to_list(self) -> List:
        return [self.x, self.y, self.z]
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

class Plane:
    def __init__(self, normal: Vec3, intercept: float):
        self.normal = normal
        self.d = intercept
    
    def __call__(self, v: Vec3) -> float:
        return np.dot(self.normal.to_numpy(), v.to_numpy()) + self.d
    
class Facet:
    def __init__(self, points: Tuple[Vec3], plane: Plane, biorthogonal: Tuple[Vec3]):
        self.points       = points
        self.plane        = plane
        self.biorthogonal = biorthogonal
    
class Quadrant:
    def __init__(self, facets: List[Facet]):
        self.facets = facets

class W:
    def __init__(self, initial_vertices:List = VERTICES):
        self.base_vertices: List = initial_vertices
        
        self.facets:    List[Facet]       = []
        self.quadrants: List[Quadrant]    = []
        self.vertices:  List[Vec3]        = []
        self.planes:    List[Plane]       = []
        self.projected_points: List[np.ndarray] = []
        self.projected_targets = {}
         
        self.__create_W()
        self.__project_points()

    def __create_W(self) -> None:
        for signs in QUADRANTS:
            s1, s2, s3 = signs
            
            vertices = [Vec3((s1 * v[0], s2 * v[1], s3 * v[2])) for v in self.base_vertices]
            
            facets = []
            for f in FACETS:
                points = (vertices[f[0] - 1], vertices[f[1] - 1], vertices[f[2] - 1])
                v1 = points[1].to_numpy() - points[0].to_numpy()
                v2 = points[2].to_numpy() - points[0].to_numpy()
                normal = np.cross(v1, v2)
                
                check_vec = CENTROID - points[0].to_numpy()
                
                normal = -normal if np.dot(normal, check_vec) > 0 else normal 
                
                unit_normal = normal / np.linalg.norm(normal)
                intercept = -np.dot(unit_normal, points[0].to_numpy())
                plane = Plane(unit_normal, intercept)
                biorthogonal = self.__calc_biorthogonal(points)
                facets.append(Facet(points, plane, biorthogonal))
                
            self.facets.extend(facets)
            self.quadrants.append(Quadrant(facets))
            self.vertices.extend(vertices)
    
    def draw_W(self, quadrants_indices=ALL_QUADRANTS_INDICES, 
               render_normals=True, 
               render_planes=True, 
               render_points=True, 
               render_cones=False,
               render_axis=True,
               origin_proj=False,
               proj=False,
               plot=True) -> plt.Axes:
        fig = plt.figure()
        ax  = plt.axes(projection="3d")
        
        if render_axis:
            self.__draw_axis(ax)
                    
        for i, q in enumerate(self.quadrants):
            if i in quadrants_indices:
                for f in q.facets:
                    verts = [p.to_list() for p in f.points]
                    poly = Poly3DCollection([verts], alpha=0.5)
                    poly.set_edgecolor("dimgray")
                    poly.set_facecolor("lightgray")
                    ax.add_collection3d(poly)
            
                    verts = np.array(verts)
        
        if render_normals:
            self.__draw_normals(ax, quadrants_indices)
        if render_planes:
            self.__draw_planes(ax, [0], [0])
        if render_cones:
            self.__draw_cone(ax, [0], [0])
        
        if render_points:
            points = np.array(POINTS)
            ax.scatter(*points[0], c="red", s=25)
            ax.scatter(*points[1], c="blue", s=25)
            ax.text(*points[0] + 0.5, "a", color="red", fontsize=12)
            ax.text(*points[1] + 0.5, "b", color="blue", fontsize=12)

            if proj:
                self.__draw_projections(ax, origin=origin_proj)
                
        ax.set_aspect('equal')
        ax.axis("off")
        if plot:
            plt.show()
        else:
            return ax
        
    def __draw_cone(self, ax: plt.Axes, quadrants: List, facets: List) -> None:
        for i, q in enumerate(self.quadrants):
            if i in quadrants:
                for j, f in enumerate(q.facets):
                    if j in facets:
                        v1, v2, v3 = [v.to_numpy() for v in f.points]
                        b1, b2, b3 = [b.to_numpy() for b in f.biorthogonal]
                        
                        ax.quiver(*CENTROID, *v1, color="black", linewidth=2, arrow_length_ratio=0.05)
                        ax.quiver(*CENTROID, *v2, color="black", linewidth=2, arrow_length_ratio=0.05)
                        ax.quiver(*CENTROID, *v3, color="black", linewidth=2, arrow_length_ratio=0.05)
                        
                        p = np.array(POINTS[0])
                        
                        a_star = [np.dot(b1, p), np.dot(b2, p), np.dot(b3, p)]

                        ax.scatter(*a_star, c="green")
                        ax.text(*a_star + 0.5, s="a*", color="red", fontsize=12)
                        
                        ax.quiver(*CENTROID, *b1 * 10, color="red", linewidth=2, arrow_length_ratio=0.05)
                        ax.quiver(*CENTROID, *b2 * 10, color="red", linewidth=2, arrow_length_ratio=0.05)
                        ax.quiver(*CENTROID, *b3 * 10, color="red", linewidth=2, arrow_length_ratio=0.05)

    def __project_points(self) -> None:
        a = np.array(POINTS[0])
        a_translated = np.array([abs(p) for p in a])
        
        b = np.array(POINTS[1])
        b_translated = np.array([abs(p) for p in b])
        
        c_translated = a_translated + b_translated
        
        for f in self.quadrants[0].facets:
            b1, b2, b3 = [b.to_numpy() for b in f.biorthogonal]
            
            a_star = np.array([np.dot(a_translated, b1), np.dot(a_translated, b2), np.dot(a_translated, b3)])
            b_star = np.array([np.dot(b_translated, b1), np.dot(b_translated, b2), np.dot(b_translated, b3)])
            c_star = np.array([np.dot(c_translated, b1), np.dot(c_translated, b2), np.dot(c_translated, b3)])
            self.projected_points.append((a_star, b_star, c_star))
        
    def __calc_biorthogonal(self, basis: np.ndarray) -> Tuple[Vec3]:
        v1, v2, v3 = [v.to_numpy() for v in basis]
        b1 = np.cross(v2, v3)
        b2 = np.cross(v1, v3)
        b3 = np.cross(v1, v2)
        
        b1 = b1 / np.dot(b1, v1)
        b2 = b2 / np.dot(b2, v2)
        b3 = b3 / np.dot(b3, v3)
        
        return (Vec3((b1[0], b1[1], b1[2])), 
                Vec3((b2[0], b2[1], b2[2])), 
                Vec3((b3[0], b3[1], b3[2])))
    
    def __draw_projections(self, ax: plt.Axes, origin: bool = False) -> None:
        if not origin:
            a, b = [np.array(list(map(abs, v))) for v in POINTS]
            
            a_orth, a_proj, l_a = self.projected_targets["a"]
            b_orth, b_proj, l_b = self.projected_targets["b"]
            
            a_shifted = a / l_a
            b_shifted = b / l_b
                
            ax.quiver(*a_shifted, *a - a_shifted, color="black", linewidth=2, arrow_length_ratio=0.0, alpha=0.8)
            ax.scatter(*a, c="red")
            ax.text(*a + 0.5, "~a", fontsize=12, c="red")

            ax.quiver(*b_shifted, *b - b_shifted, color="black", linewidth=2, arrow_length_ratio=0.0, alpha=0.8)
            ax.scatter(*b, c="blue")
            ax.text(*b + 0.5, "~b", fontsize=12, c="blue")
            
            ax.scatter(*a_shifted, c="red")
            a_shifted[2] += 1.0
            ax.text(*a_shifted + 0.5, "a*", fontsize=12, c="red")
            
            ax.scatter(*b_shifted, c="blue")
            b_shifted[2] -= 1.0
            b_shifted[1] += 1.0
            ax.text(*b_shifted, "b*", fontsize=12, c="blue")    
        else:
            a, b = [np.array(v) for v in POINTS]
            
            a_orth, a_proj, l_a = self.projected_targets["a"]
            b_orth, b_proj, l_b = self.projected_targets["b"]
            
            a_shifted = a / l_a
            b_shifted = b / l_b
            
            ax.quiver(*a_shifted, *a - a_shifted, color="black", linewidth=2, arrow_length_ratio=0.0, alpha=0.8)
            ax.quiver(*b_shifted, *b - b_shifted, color="black", linewidth=2, arrow_length_ratio=0.0, alpha=0.8)
             
            ax.scatter(*a_shifted, c="red")
            a_shifted[2] += 1.0
            a_shifted[1] -= 0.5
            ax.text(*a_shifted, "a*", fontsize=12, c="red")
            ax.scatter(*b_shifted, c="blue")
            b_shifted[2] += 1.0
            ax.text(*b_shifted, "b*", fontsize=12, c="blue")
            
    def __draw_axis(self, ax: plt.Axes, label_axis=False) -> None:
        axis_length = 20
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']
        
        for i in range(3):
            direction = [0, 0, 0]
            direction[i] = axis_length
            neg_direction = [-d for d in direction]
            ax.quiver(*CENTROID, *neg_direction, color=colors[i], linewidth=2, arrow_length_ratio=0.0, alpha=0.8)
            
            ax.quiver(*CENTROID, *direction, color=colors[i], linewidth=2, arrow_length_ratio=0.1, alpha=0.8)
            
            if label_axis:
                label_pos = [0, 0, 0]
                label_pos[i] = axis_length * 1.1
                ax.text(*label_pos, labels[i], color=colors[i], fontsize=24, fontweight='bold')
        
    def __draw_normals(self, ax: plt.Axes, quadrants=ALL_QUADRANTS_INDICES) -> None:
        normals = []
        for idx, q in enumerate(self.quadrants):
            if idx in quadrants:
                for f in q.facets:
                    normal = f.plane.normal
                    center = np.mean([p.to_numpy() for p in f.points], axis=0)
                    
                    normals.append((normal, center))
        
        centers = np.array([n[1] for n in normals])
        vectors = np.array([n[0] for n in normals])
        
        ax.quiver(
            centers[:, 0], centers[:, 1], centers[:, 2],
            vectors[:, 0], vectors[:, 1], vectors[:, 2],
            length=1.0,
            color="black",
            arrow_length_ratio=0.5,
            linewidth=2
        )
         
    def __draw_planes(self, ax: plt.Axes, quadrants=ALL_QUADRANTS_INDICES, facets=None) -> None:
        q_counter = 0
        f_counter = 0
        for q in self.quadrants:
            if q_counter in quadrants:
                for f in q.facets:
                    if facets is None or f_counter in facets:
                        nx, ny, nz = f.plane.normal
                        d = f.plane.d
                        
                        x = np.linspace(-10, 10, 20)
                        y = np.linspace(-10, 10, 20)
                        X, Y = np.meshgrid(x, y)
                        
                        Z = -(nx * X + ny * Y + d) / nz
                        ax.plot_surface(X, Y, Z, color="gray", alpha=0.5)
                        f_counter += 1
                q_counter += 1

    def dump_tex_tables(self, pth: str) -> None:
        pth2vertices  = os.path.join(pth, "vertices.tex")
        pth2allvertices  = os.path.join(pth, "all_vertices.tex")
        pth2uniquevertices  = os.path.join(pth, "uniq_vertices.tex")
        pth2facets = os.path.join(pth, "facets.tex")
        pth2quadrants = os.path.join(pth, "quadrants.tex")
        pth2proj = os.path.join(pth, "proj.tex")
        
        self.__dump_base_vertices(pth2vertices)
        self.__dump_all_vertices(pth2allvertices)
        self.__dump_unique_vertices(pth2uniquevertices)
        self.__dump_facets(pth2facets)
        self.__dump_proj(pth2proj)
    
    def __dump_base_vertices(self, pth: str) -> None:
        with open(pth, 'w+', encoding='utf-8') as f:
            f.write(r'\begin{table}[h]' + '\n')
            f.write(r'    \centering' + '\n')
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
            f.write(r'    \caption{Вершины первого квадранта}' + '\n')
            f.write(r'\end{table}' + '\n')
    
    def __fraction_format(self, value: float, numerator:int = 33, denominator:int = 5, tol: float = 1e-9) -> str:
        target = numerator / denominator
        if abs(value - target) < tol:
            return r"$\frac{" + str(numerator) + "}{" + str(denominator) + "}$"
        elif abs(value + target) < tol:
            return r"$-\frac{" + str(numerator) + "}{" + str(denominator) + "}$"
        return str(value)
        
    def __dump_all_vertices(self, pth: str) -> None:
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
                    
                    x = self.__fraction_format(v.x, 33, 5)
                    y = self.__fraction_format(v.y, 171, 32)
                    z = self.__fraction_format(v.z, 107, 13)
                    
                    row = f"            {j + 1} & {x} & {y} & {z} \\\\\n"
                    f.write(row)
                
                f.write(r"        \end{tabular}" + "\n")
                f.write(r"    \end{subtable}" + "\n")
                if j != 3:
                    f.write(r"    \hfill" + "\n")

                l += 16
                r += 16
            f.write(r"    \caption{Все вершины многогранника}" + "\n")
            f.write(r"\end{table}" + "\n")
            
    def __dump_unique_vertices(self, pth: str) -> None:
        unique_vertices = []
        seen_coords = set()
        
        for v in self.vertices:
            x = self.__fraction_format(v.x, 33, 5)
            y = self.__fraction_format(v.y, 171, 32)
            z = self.__fraction_format(v.z, 107, 13)
            
            if (x, y, z) not in seen_coords:
                seen_coords.add((x, y, z))
                unique_vertices.append((x, y, z))

        mid_point = len(unique_vertices) // 2
        parts = [unique_vertices[:mid_point], unique_vertices[mid_point:]]

        with open(pth, 'w+', encoding='utf-8') as f:
            f.write(r'\begin{table}[h]' + '\n')
            f.write(r'    \centering' + '\n')

            for i, part in enumerate(parts):
                if not part and i > 0:
                    continue

                f.write(r'    \begin{subtable}{0.45\textwidth}' + '\n')
                f.write(r'        \centering' + '\n')
                f.write(r'        \begin{tabular}{c||ccc}' + '\n')
                f.write(r'            \toprule' + '\n')
                f.write(r'            \textbf{№} & \textbf{$x$} & \textbf{$y$} & \textbf{$z$} \\'+ '\n')
                f.write(r'            \midrule' + '\n') 
                
                start_num = 0 if i == 0 else mid_point

                for k, (x, y, z) in enumerate(part):
                    row_num = start_num + k + 1
                    f.write(f"            {row_num} & {x} & {y} & {z} \\\\\n")
                
                f.write(r'            \bottomrule' + '\n')
                f.write(r'        \end{tabular}' + '\n')
                f.write(r'    \end{subtable}' + '\n')

            f.write(r'    \caption{Уникальные вершины многогранника}' + '\n')
            f.write(r'\end{table}' + '\n')
   
    def __dump_facets(self, pth: str) -> None:
        with open(pth, 'w+', encoding='utf-8') as f:
            f.write(r'\begin{table}[h]' + '\n')
            f.write(r'    \centering' + '\n')
            f.write(r'    \begin{subtable}{0.45\textwidth}' + '\n')
            f.write(r'        \centering' + '\n')
            f.write(r'        \begin{tabular}{c||ccc}' + '\n')
            f.write(r'            \toprule' + '\n')
            f.write(r'            \textbf{№} & \textbf{$v_1$} & \textbf{$v_2$} & \textbf{$v_3$} \\'+ '\n')
            f.write(r'            \midrule' + '\n') 
            for idx, part in enumerate(FACETS, start=1):
                i, j, k = part
                f.write(f"            {idx} & {i} & {j} & {k} \\\\\n")
                
            f.write(r'            \bottomrule' + '\n')
            f.write(r'        \end{tabular}' + '\n')
            f.write(r'    \end{subtable}' + '\n')
            f.write(r'    \caption{Правила построения граней в квадрантах}' + '\n')
            f.write(r'\end{table}' + '\n')

    def __dump_proj(self, pth: str) -> None:
        with open(pth, "w+", encoding='utf-8') as f:
            f.write(r'\begin{table}[h]' + '\n')
            f.write(r'    \centering' + '\n')
            f.write(r'    \begin{subtable}{0.45\textwidth}' + '\n')
            f.write(r'        \centering' + '\n')
            f.write(r'        \begin{tabular}{c||ccccc}' + '\n')
            f.write(r'            \toprule' + '\n')
            f.write(r'            \textbf{Грань} & \textbf{$x_1$} & \textbf{$x_2$} & \textbf{$x_3$} & \textbf{$\sum$} & \textbf{$\lambda$} \\'+ '\n')
            f.write(r'            \midrule' + '\n') 
            
            a, b = [np.array(list(map(abs, p))) for p in POINTS]
            
            for i, (a, b, c) in enumerate(self.projected_points):
                l_a = sum(a)
                l_b = sum(b)
                l_c = sum(c)
                
                proj_a = a / l_a
                proj_b = b / l_b
                proj_c = c / l_c
                
                valid_a = all(v > 0 for v in a)
                valid_b = all(v > 0 for v in b)
                valid_c = all(v > 0 for v in c)
                
                if valid_a: self.projected_targets["a"] = (a, proj_a, l_a)
                if valid_b: self.projected_targets["b"] = (b, proj_b, l_b)
                if valid_c: self.projected_targets["c"] = (c, proj_c, l_c)
                
                row_color_a = r'\rowcolor{green!20}' if valid_a else ''
                row_color_b = r'\rowcolor{green!20}' if valid_b else ''
                row_color_c = r'\rowcolor{green!20}' if valid_c else ''
                
                f.write(f"            {row_color_a}\n")
                f.write(f"            $a_{{{i+1}}}$ & {proj_a[0]:.2f} & {proj_a[1]:.2f} & {proj_a[2]:.2f} & {np.sum(proj_a):.2f} & {np.sum(l_a):.2f} \\\\\n")
                
                f.write(f"            {row_color_b}\n")
                f.write(f"            $b_{{{i+1}}}$ & {proj_b[0]:.2f} & {proj_b[1]:.2f} & {proj_b[2]:.2f} & {np.sum(proj_b):.2f} & {np.sum(l_b):.2f} \\\\\n")
                
                f.write(f"            {row_color_c}\n")
                f.write(f"            $c_{{{i+1}}}$ & {proj_c[0]:.2f} & {proj_c[1]:.2f} & {proj_c[2]:.2f} & {np.sum(proj_c):.2f} & {np.sum(l_c):.2f} \\\\\n")
                
                f.write(r'            \midrule' + '\n')
                    
            f.write(r'            \bottomrule' + '\n')
            f.write(r'        \end{tabular}' + '\n')
            f.write(r'    \end{subtable}' + '\n')
            f.write(r'    \caption{Точки в биортогональном базисе в первом квадранте}' + '\n')
            f.write(r'\end{table}' + '\n')
            
def generate_tex(cover: W) -> None:
    cover.dump_tex_tables(REPORT_PTH)
    
    ax = cover.draw_W(quadrants_indices=[0], render_normals=False, render_planes=False, render_points=False, plot=False)
    
    ax.view_init(10, 40, 0)
    plt.savefig(os.path.join(IMAGES_PTH, "quadrant1.png"), dpi=1000, bbox_inches='tight')
    
    ax = cover.draw_W(render_normals=False, render_planes=False, render_points=False, plot=False)
    ax.view_init(25, 35, 0)
    
    ax = cover.draw_W(render_normals=True, render_planes=False, render_points=False, plot=False)
    ax.view_init(0, 45, 0)
    plt.savefig(os.path.join(IMAGES_PTH, "polytope-normals.png"), dpi=1000, bbox_inches='tight')
    
    ax = cover.draw_W(render_normals=True, render_planes=True, render_points=False, plot=False)
    ax.view_init(10, 135, 0)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_zlim(0, 15)
    plt.savefig(os.path.join(IMAGES_PTH, "polytope-plane.png"), dpi=1000, bbox_inches='tight')
    
    ax = cover.draw_W(quadrants_indices=ALL_QUADRANTS_INDICES, render_points=True, render_cones=True, render_axis=False, render_planes=False, render_normals=False, plot=False)
    ax.view_init(20, 70, 0)
    plt.savefig(os.path.join(IMAGES_PTH, "polytope-cone.png"), dpi=1000, bbox_inches='tight')
    
    ax = cover.draw_W(quadrants_indices=ALL_QUADRANTS_INDICES, proj=True, render_points=True, render_cones=False, render_axis=True, render_planes=False, render_normals=False, plot=False)
    ax.view_init(20, -20, 0)
    plt.savefig(os.path.join(IMAGES_PTH, "polytope-posproj.png"), dpi=1000, bbox_inches='tight')
    
    ax = cover.draw_W(quadrants_indices=ALL_QUADRANTS_INDICES, proj=True, origin_proj=True, render_points=True, render_cones=False, render_axis=True, render_planes=False, render_normals=False, plot=False)
    ax.view_init(20, 25, 0)
    plt.savefig(os.path.join(IMAGES_PTH, "polytope-originproj.png"), dpi=1000, bbox_inches='tight')
    
def main() -> None:
    cover = W()
    generate_tex(cover)
    
    # cover.draw_W(render_normals=False, render_planes=False)


if __name__ == "__main__":
    main()