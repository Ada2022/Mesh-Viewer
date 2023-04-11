import numpy as np
import math
import pygame


class Reader:
    def __init__(self) -> None:
        pass

    def read_object_file(self, file_name):
        # Define variables for object data
        vertices = {}
        faces = []

        # Read object file and extract vertices and faces
        with open(file_name, 'r') as f:
            vertex_count, face_count = map(int, f.readline().split(','))
            for i in range(vertex_count):
                vertex_id, x, y, z = map(float, f.readline().split(','))
                vertices[int(vertex_id)] = (x, y, z)
            for i in range(face_count):
                v1, v2, v3 = map(int, f.readline().split(','))
                faces.append((v1, v2, v3))

        return vertices, faces


class Object:
    def __init__(self, vertices, faces, projected_vertices={}, scaled_vertices={}) -> None:
        self.vertices = vertices
        self.faces = faces
        self.projected_vertices = projected_vertices
        self.scaled_vertices = scaled_vertices
        self.scale = 0

    def orthographic_projection(self):
        # Define projection matrix
        proj_matrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 1]])

        self.projected_vertices = {}
        # Define projection function
        for key, value in self.vertices.items():
            vertex = np.array(value + (1,))
            proj = np.dot(proj_matrix, vertex)[:2]
            self.projected_vertices[key] = proj

        # Apply projection to each vertex and output            

    def get_scale(self, height, width):
        
        vertices_array = list(self.vertices.values())
        vertices_array = np.array(vertices_array )
        x_diff = np.abs(vertices_array[:, 0].max() - vertices_array[:, 0].min())
        y_diff = np.abs(vertices_array[:, 1].max() - vertices_array[:, 1].min())
        
        scale = max(height/2, width/2) / max(x_diff, y_diff)
        return scale

    def scale_object(self, height, width):
        if self.scale == 0: self.scale = self.get_scale(height, width)
        # Define transformation matrices
        transformation_matrix = np.array([[self.scale, 0, width / 2], 
                                        [0, -self.scale, height / 2], 
                                        [0, 0, 1]])

        # Scale and translate each vertex
        self.scaled_vertices = {}
        for key, value in self.projected_vertices.items():
            # Convert to homogeneous coordinates
            vertex = np.array(tuple(value) + (1,))

            # Apply transformation matrices
            transformed_vertex = np.dot(transformation_matrix, vertex)

            # Convert back to 2D coordinates and add to dictionary
            self.scaled_vertices[key] = transformed_vertex[:2]

    def rotate_object(self, vector_x, vector_y, angle):
        # Convert angle from degrees to radians
        vector_z = 0
        radians = angle % (3.14159)
        
        # Compute sin and cosine of the angle
        c = np.cos(radians)
        s = np.sin(radians)

        # Create the 3x3 rotation matrix for 3D rotation
        # based on the given axis vector
        norm = np.sqrt(vector_x**2 + vector_y**2 + vector_z**2)
        ux = vector_x / norm
        uy = vector_y / norm
        uz = vector_z / norm
        rotation_matrix = np.array([
            [c + ux**2 * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s, 0],
            [ux * uy * (1 - c) + uz * s, c + uy**2 * (1 - c), uy * uz * (1 - c) - ux * s, 0],
            [ux * uz * (1 - c) - uy * s, uy * uz * (1 - c) + ux * s, c + uz**2 * (1 - c), 0],
            [0, 0, 0, 1]
        ])
        
        # print(rotation_matrix)
        # Rotate each vertex
        for key, value in self.vertices.items():
            # Create the 3x1 homogeneous vector for vertex
            vertex = np.array([value[0], value[1], value[2], 1])
            
            # Compute the rotated vertex
            rotated_vertex = np.dot(rotation_matrix, vertex)

            # Store the rotated vertex back into self.vertices
            self.vertices[key] = tuple(rotated_vertex[:3])


class MeshViewer: 
    def __init__(self, width, height, file_name, fill = True): 
        self.width = width 
        self.height = height 
        self.file_name = file_name
        self.obj = None
        self.running = False
        self.left_pressed = False
        self.start_pos = None
        self.last_pos = None
        self.fill = fill

    def get_normal(self, p1, p2, p3):
        # convert points to numpy arrays
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        
        # compute two vectors on the plane
        v1 = p2 - p1
        v2 = p3 - p1
        
        # compute the cross product of the vectors
        normal = np.cross(v1, v2)
        
        # normalize the normal vector to unit length
        normal = normal / np.linalg.norm(normal)
        
        return normal
    
    def interpolate_color(self, angle):
        r1, g1, b1 = 0, 0, 95  # #00005F
        r2, g2, b2 = 0, 0, 255  # #0000FF
        t = angle / (math.pi/2)
        r = int(r1 + t * (r2 - r1))
        g = int(g1 + t * (g2 - g1))
        b = int(b1 + t * (b2 - b1))
        return (r, g, b)

    def run(self):
        pygame.init()
        self.running = True
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Mesh Viewer')
        clock = pygame.time.Clock()

        reader = Reader()
        vertices, faces = reader.read_object_file(self.file_name)
        self.obj = Object(vertices, faces)
        self.obj.orthographic_projection()
        self.obj.scale_object(self.height, self.width)


        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.left_pressed = True
                    self.start_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.left_pressed = False
                    self.last_pos = None
                elif event.type == pygame.MOUSEMOTION and self.left_pressed:
                    if self.start_pos is not None:
                        end_pos = pygame.mouse.get_pos()
                        distance = ((end_pos[0] - self.start_pos[0]) ** 2 + (end_pos[1] - self.start_pos[1]) ** 2) ** 0.5
                        if distance > 0:
                            vector_x = (end_pos[1] - self.start_pos[1]) / distance
                            vector_y = (end_pos[0] - self.start_pos[0]) / distance
                            self.obj.rotate_object(vector_x, vector_y, distance / 200)
                            self.start_pos = end_pos

            self.obj.orthographic_projection()
            self.obj.scale_object(self.height, self.width)

            screen.fill(WHITE)

            for vertex in self.obj.scaled_vertices.values():
                pygame.draw.circle(screen, BLUE, vertex, 5)


            sorted_faces = sorted(self.obj.faces, key=lambda face: np.mean([self.obj.vertices[i][2] for i in face])) 
            # draw faces
            for face in sorted_faces:
                vertices_list = [self.obj.scaled_vertices[i] for i in face]
                
                if self.fill:
                    normal = self.get_normal(self.obj.vertices[face[0]], self.obj.vertices[face[1]], self.obj.vertices[face[2]])
                    angle = math.acos(normal[0]) / math.pi
                    color = self.interpolate_color(angle)

                    pygame.draw.polygon(screen, color, vertices_list, 0)
                    pygame.draw.polygon(screen, PURPLE, vertices_list, 2)
                else:
                    pygame.draw.polygon(screen, BLUE, vertices_list, 1)
            
            # update the screen
            pygame.display.flip()
            
            # set the frame rate
            clock.tick(60)
        
        # quit pygame
        pygame.quit()


if __name__ == '__main__':
    # set up colors
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    PURPLE = (255, 0, 255)
    # set up the windows
    screen_width = 800
    screen_height = 600
    viewer = MeshViewer(screen_width, screen_height, '../dataset/object.txt')
    viewer.run()


