import pygame

class Visualize:
    def __init__(self, gnome, display_size = (500, 500)):
        self.gnome = gnome
        self.display_size = display_size
        pygame.init()
        self.surface = pygame.display.set_mode(self.display_size)
        pygame.display.set_caption('NEAT GNOME')``
    
    def draw_node(self, location):
        pygame.draw.circle(self.surface, (255, 255, 255), location, 50)

    def draw(self):
        self.draw_node((60 , 60))
        pygame.display.update()

if __name__ == "__main__":
    viz = Visualize(None)
    viz.draw()