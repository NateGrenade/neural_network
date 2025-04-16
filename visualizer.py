from model_generator import MODEL_PATH
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import pygame
import sys
from model_generator import train

# Constants
PREDICTED_DIGIT = None
layer_input = "2f"
nodes_input = "128, 128"
input_active = None
layer_total = 3
node_list = [128, 128]
activations_2d = []
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

class NodeVis:
    def __init__(self, activation):
        self.activation = activation
        self.color = (255-(activation*255), activation*255, 0)
    def draw(self, surface, center):
        r, g, b = (255-(self.activation*255), self.activation*255, 0)
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        self.color = (r, g, b)
        pygame.draw.circle(surface, self.color, center, 8)

class Layer:
    def __init__(self, layer_num, output=False):
        self.layer_num = layer_num
        self.nodes = []
        self.output = output
    def add_node(self, node):
        self.nodes.append(node)
    def draw_layer(self):
        delta = (HEIGHT-100)/(len(self.nodes))
        for i in range(len(self.nodes)):
            x = self.layer_num * (WIDTH - 500) / layer_total + 400
            y = 100 + (delta * i)
            self.nodes[i].draw(screen, (x, y))
            if self.output:
                label_text = font.render(str(i), True, BLACK)
                label_pos = (x + 30, y - label_text.get_height() // 2)
                screen.blit(label_text, label_pos)
    def modify_layer(self, activations):
        print(f"Updating layer {self.layer_num} with {len(activations)} activations")
        for i, node in enumerate(self.nodes):
            node.activation = activations[i] if i < len(activations) else 0
        self.draw_layer()

class Button:
    def __init__(self, x, y, width, height, text, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.color = LIGHT_GRAY
    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        text_surf = font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
    def check_click(self, pos):
        if self.rect.collidepoint(pos):
            self.callback()

def on_button_click():
    global model, PREDICTED_DIGIT, activations_2d
    if model is None:
        print("error: No model loaded")
        return
    try:
        canvas_str = pygame.image.tostring(canvas_surface, "RGB")
        image = Image.frombytes("RGB", (280, 280), canvas_str)
        image = image.convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        image_arr = np.array(image) / 255.0
        image_arr = image_arr.reshape(1, 28, 28)
        # Get activations layer by layer
        activations_2d.clear()
        x = image_arr
        for layer in model.layers:
            # Skip InputLayer or other non-callable layers
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            try:
                x = layer(x)
                # Only store activations from Dense or output layers
                if isinstance(layer, tf.keras.layers.Dense):
                    activations_2d.append(x[0].numpy())
            except Exception as e:
                print(f"Error on layer {layer.name}: {e}")
        print(f"Stored {len(activations_2d)} layer activations")
        for i, layer in enumerate(layers):
            if i < len(activations_2d):
                layer.modify_layer(activations_2d[i])
        PREDICTED_DIGIT = np.argmax(activations_2d[-1])
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
        if len(new_nodes) != new_layers - 1:
            print("Err: list length of nodes must match number of hidden layers")
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
        output = Layer(layer_total, True)
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
        screen = pygame.display.set_mode((800, 600))

buttons = [
    Button(100, 750, 150, 50, "Predict Digit", on_button_click),
    Button(100, 850, 150, 50, "Retrain Model", retrain_model),
    Button(100, 950, 150, 50, "Exit", lambda: sys.exit())
]
layer_rect = pygame.Rect(50, 50, 200, 30)
nodes_rect = pygame.Rect(50, 100, 200, 30)

try:
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    is_fullscreen = True
    pygame.display.set_caption("Interactive Neural Network")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    canvas_rect = pygame.Rect(50, 300, 280, 280)
    canvas_surface = pygame.Surface((280, 280))
    canvas_surface.fill(WHITE)
    drawing = False
    layers = []
    for k in range(layer_total-1):
        layer = Layer(k+1)
        for _ in range(node_list[k]):
            layer.add_node(NodeVis(0))
        layers.append(layer)
    output = Layer(layer_total, True)
    for i in range(10):
        output.add_node(NodeVis(0))
    layers.append(output)
    print(f"Initialized {len(layers)} layers")
except Exception as e:
    print(f"yikers... {e}")
    sys.exit(-1)

running = True
try:
    while running:
        screen.fill(BACKGROUND)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
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
                    input_active = "layers"
                elif nodes_rect.collidepoint(event.pos):
                    input_active = "nodes"
                elif canvas_rect.collidepoint(event.pos):
                    drawing = True
                    input_active = None
                else:
                    input_active = None
                if event.button == 1:
                    for button in buttons:
                        button.check_click(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                if canvas_rect.collidepoint(event.pos):
                    pygame.draw.circle(canvas_surface, BLACK, (event.pos[0] - canvas_rect.x, event.pos[1] - canvas_rect.y), 8)
        screen.blit(canvas_surface, canvas_rect.topleft)
        pygame.draw.rect(screen, BLACK, canvas_rect, 2)
        for button in buttons:
            button.draw(screen)
        for layer in layers:
            layer.draw_layer()
        text = "Draw a digit and predict!" if PREDICTED_DIGIT is None else f"Predicted: {PREDICTED_DIGIT}"
        text_surf = font.render(text, True, BLACK)
        screen.blit(text_surf, (50, 250))
        pygame.draw.rect(screen, WHITE if input_active == "layers" else LIGHT_GRAY, layer_rect)
        pygame.draw.rect(screen, BLACK, layer_rect, 2)
        layer_surf = font.render(f"Layers: {layer_input}", True, BLACK)
        screen.blit(layer_surf, (layer_rect.x+5, layer_rect.y+5))
        pygame.draw.rect(screen, WHITE if input_active == "nodes" else LIGHT_GRAY, nodes_rect)
        pygame.draw.rect(screen, BLACK, nodes_rect, 2)
        nodes_surf = font.render(f"Nodes: {nodes_input}", True, BLACK)
        screen.blit(nodes_surf, (nodes_rect.x+5, nodes_rect.y+5))
        pygame.display.flip()
        clock.tick(60)
except Exception as e:
    print(f"Yikers... {e}")
    running = False

pygame.quit()
sys.exit()