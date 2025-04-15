# _____________ imports _____________
from model_generator import MODEL_PATH
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import pygame
import sys
from model_generator import train


# _____________ constants/initializing _____________
PREDICTED_DIGIT = None
layer_input = "2"
nodes_input = "128,128"
input_active = None
# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Yikers... {e}")
    model = None


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND = (150, 150, 200)
LIGHT_GRAY = (200, 200, 200)

# _____________ classes _____________

class NodeVis:
    def __init__(self, activation):
        self.activation = activation
        self.color = (255-(activation*255), activation*255, 0)
    def draw(self, surface, center):
        pygame.draw.circle(surface, self.color, center, 8)

class Layer:
    def __init__(self, layer_num, output = False):
        self.layer_num = layer_num
        self.nodes = []
        self.output = output

    def add_node(self, node):
        self.nodes.append(node)

    def draw_layer(self):
        delta = (HEIGHT-100)/(self.nodes.__len__())
        for i in range(self.nodes.__len__()):
            x = self.layer_num * (WIDTH - 500) / layer_total + 400
            y = 100 + (delta * i)
            self.nodes[i].draw(screen, (x, y))
            if self.output:
                label_text = font.render(str(i), True, (0, 0, 0))
                label_pos = (x + 30, y - label_text.get_height() // 2)
                screen.blit(label_text, label_pos)

    def modify_layer(self, activations):
        print(f"Updating layer {self.layer_num} with {len(activations[0])}")
        for i, node in enumerate(self.nodes):
            node.activation = activations[0][i] if i < len(activations[0]) else 0
            node.color = (255-(node.activation*255), node.activation*255, 0)

# Button Class
class Button:
    def __init__(self, x, y, width, height, text, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.color = LIGHT_GRAY

    def draw(self, surface):
        # Draw button rectangle
        pygame.draw.rect(surface, self.color, self.rect)
        # Draw text centered in button
        text_surf = font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_click(self, pos):
        # If mouse position is within the button rect, run callback
        if self.rect.collidepoint(pos):
            self.callback()

# Example callback function
def on_button_click():
    global model, PREDICTED_DIGIT
    if model is None:
            print("error: No model loaded")
            return
    try:
        print("Processing canvas image...")
        canvas_str = pygame.image.tostring(canvas_surface, "RGB")
        image = Image.frombytes("RGB", (280, 280), canvas_str)
        # Convert to grayscale and resize
        image = image.convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        image_arr = np.array(image) / 255.0
        image_arr = image_arr.reshape(1, 28, 28)
        print("Predicting Digit...")
        # Load model
        prediction = model.predict(image_arr)
        PREDICTED_DIGIT = np.argmax(prediction)
        print(f"Predicted digit: {PREDICTED_DIGIT}")
        canvas_surface.fill(WHITE)
    except Exception as e:
        print(f"yikers... {e}")



def retrain_model():
    global model, layer_total, node_list, layers
    try:
        print("Retraining model")
        new_layers = int(layer_input) + 1 if layer_input.isdigit() else layer_total
        new_nodes = [int(n) for n in nodes_input.split(",") if n.strip().isdigit()]
        if len(new_nodes) != new_layers -1:
            print("Err: node count must match hidden layers")
            return
        layer_total = new_layers
        node_list = new_nodes
        train(layer_total, node_list)
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model retrained and reloaded")

        layers = []
        for k in range(layer_total-1):
            layer = Layer(k+1)
            for _ in range(node_list[k]):
                layer.add_node(NodeVis(0))
                layers.append(layer)
            ouput = Layer(layer_total, True)
            for i in range(10):
                output.add_node(NodeVis(0))
            layers.append(output)
            print(f"Updated {len(layers)} layers")
    except Exception as e:
        print(f"Yikers: {e}")




def toggle_fullscreen():
    global is_fullscreen, screen
    is_fullscreen = not is_fullscreen
    if is_fullscreen:
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((800, 600))  # Go back to windowed mode


# UI Buttons
buttons = [
    Button(100, 750, 150, 50, "Predict Digit", on_button_click),
    Button(100, 850, 150, 50, "Retrain Model", retrain_model),
    Button(100, 950, 150, 50, "Exit", lambda: sys.exit())
]
layer_rect = pygame.Rect(50, 50, 200, 30)
nodes_rect = pygame.Rect(50, 100, 200, 30)


# _____________ app logic _____________
"""
try:
    print("<Welcome to the Interactive Neural Network>")
    print("Please fill in all information before opening the application")
    layer_total = int(input("Number of hidden layers: ")) + 1
    node_list = []
    for j in range(layer_total-1):
        nodes = int(input("Number of nodes in hidden layer #"+str(j+1)+": "))
        node_list.append(nodes)
    print(f"Set up {layer_total-1} hidden layers with nodes: {node_list}")
    retrain_model()
except Exception as e:
    print(f"Error setting up layers: {e}")
    sys.exit(-1)
"""

try:
    # Initialize pygame
    pygame.init()
    # Creates the display
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    is_fullscreen = True
    pygame.display.set_caption("Interactive Neural Network")
    # Set up "in game" clock
    clock = pygame.time.Clock()
    # font used for buttons/text
    font = pygame.font.SysFont("Arial", 24)
    canvas_rect = pygame.Rect(50, 300, 280, 280)  # Your drawing box (10x scale of 28x28)
    canvas_surface = pygame.Surface((280, 280))
    canvas_surface.fill(WHITE)
    drawing = False
    #drawing board, so to speak, for the input digit
    layers = []
    for k in range(layer_total-1):
        layer = Layer(k+1)
        for l in range(node_list[k]):
            layer.add_node(NodeVis(0))
        layers.append(layer)

    output = Layer(layer_total, True)
    for i in range(10):
        output.add_node(node = NodeVis(0))
    layers.append(output)
    print(f"Initialized {len(layers)} layers for visualization")
except Exception as e:
    print(f"yikers... {e}")
    sys.exit(-1)

running = True
try:
    while running:
        screen.fill(BACKGROUND)  # Sets the background color to white
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            #quits upon clicking the "quit" button
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                # Makes 'F' toggle fullscreen
                    toggle_fullscreen()
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif input_active == "layers":
                    if event.key == pygame.K_RETURN:
                        input_active = None
                    elif event.key == pygame.K_BACKSPACE:
                        layer_input = layer_input[:-1]
                    elif event.unicode.isdigit():
                        layer_input += event.unicode
                elif input_active == "nodes":
                    if event.key == pygame.K_RETURN:
                        input_active = None
                    elif event.key == pygame.K_BACKSPACE:
                        nodes_input = nodes_input[:-1]
                    elif event.unicode.isdigit() or event.unicode == ",":
                        nodes_input += event.unicode
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if layer_rect.collidepoint(event.pos):
                    input_active = layers
                elif nodes_rect.collidepoint(event.pos):
                    input_active = "nodes"
                elif canvas_rect.collidepoint(event.pos):
                    drawing = True
                    input_active = None
                else:
                    input_active = None
                if event.button == 1:  # Clicking with the left mouse button
                    for button in buttons:
                        button.check_click(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                if canvas_rect.collidepoint(event.pos):
                    pygame.draw.circle(canvas_surface, BLACK, (event.pos[0] - canvas_rect.x, event.pos[1] - canvas_rect.y),
                                       8)
        # Draw the canvas area to the screen
        screen.blit(canvas_surface, canvas_rect.topleft)
        pygame.draw.rect(screen, BLACK, canvas_rect, 2)  # Outline
        # Draw all buttons
        for button in buttons:
            button.draw(screen)
        for layer in layers:
            layer.draw_layer()

        if PREDICTED_DIGIT is None:
                text_surf = font.render(f"Draw a digit and see what the model predicts!", True, BLACK)
                screen.blit(text_surf, (50, 250))
        if PREDICTED_DIGIT is not None:
                text_surf = font.render(f"Predicted: {PREDICTED_DIGIT}", True, BLACK)
                screen.blit(text_surf, (50, 250))

        pygame.draw.rect(screen, WHITE if input_active == "layers" else LIGHT_GRAY, layer_rect)
        pygame.draw.rect(screen, BLACK, layer_rect, 2)
        layer_surf = font.render(f"Layers: {layer_input}", True, BLACK)
        screen.blit(layer_surf, (layer_rect.x+5, layer_rect.y + 5))
        pygame.draw.rect(screen, WHITE if input_active == "nodes" else LIGHT_GRAY, nodes_rect)
        pygame.draw.rect(screen, BLACK, nodes_rect, 2)
        nodes_surf = font.render(f"Nodes: {nodes_input}", True, BLACK)
        screen.blit(nodes_surf, (nodes_rect.x + 5, nodes_rect.y + 5))

        pygame.display.flip()  # Update display
        clock.tick(60)  # Limit to 60 FPS
except Exception as e:
    print(f"Yikers... {e}")
    running = False

pygame.quit()
sys.exit()