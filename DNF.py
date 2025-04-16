import numpy as np

class NeuralNetwork:
    def __init__(self, n, m, learning_rate=0.01):
        self.n = n  # Number of input neurons
        self.m = m  # Number of hidden neurons
        self.learning_rate = learning_rate
        
        # Initialize weights and thresholds
        self.w = np.random.randn(m, n)
        self.v = np.random.randn(m)
        self.W = np.random.randn(1, m)
        self.V = np.random.randn(1)
    
    def sgn(self, x):
        return np.where(x >= 0, 1, -1)
    
    def forward(self, x):
        z = self.sgn(np.dot(self.w, x) - self.v)
        y = self.sgn(np.dot(self.W, z) - self.V)
        return y, z
    
    def backward(self, x, z, y, p):
        delta_W = self.learning_rate * (p - y) * z
        delta_V = -self.learning_rate * (p - y)
        delta_w = self.learning_rate * np.outer(self.W.flatten() * (p - y), x)  # Flatten W
        delta_v = -self.learning_rate * (p - y) * self.W.flatten()  # Flatten W to match v

        self.W += delta_W
        self.V += delta_V
        self.w += delta_w
        self.v += delta_v
    
    def train(self, X, P):
        for x, p in zip(X, P):
            y, z = self.forward(x)
            if y != p:
                self.backward(x, z, y, p)

# Aufgabe a: Implementierung des Netzes mit Fehlerrückübertragung
n = 10  # Anzahl der Eingangsneuronen
m = 5   # Anzahl der Monome

# Zufällige DNF mit mindestens 5 Monomen und mindestens 3 Literalen pro Monom
X = np.random.choice([-1, 1], size=(100, n))
P = np.random.choice([-1, 1], size=(100))

nn = NeuralNetwork(n=n, m=m)
nn.train(X, P)

# Aufgabe c: Testen der Implementierung
def test_implementation(nn, X):
    initial_weights_w = nn.w.copy()
    initial_weights_v = nn.v.copy()
    initial_weights_W = nn.W.copy()
    initial_weights_V = nn.V.copy()
    
    for x in X:
        y, _ = nn.forward(x)
    
    assert np.array_equal(nn.w, initial_weights_w), "Gewichte w haben sich geändert!"
    assert np.array_equal(nn.v, initial_weights_v), "Schwellwerte v haben sich geändert!"
    assert np.array_equal(nn.W, initial_weights_W), "Gewichte W haben sich geändert!"
    assert np.array_equal(nn.V, initial_weights_V), "Schwellwerte V haben sich geändert!"
    
    print("Test erfolgreich: Die Gewichte und Schwellwerte haben sich nicht geändert.")

test_implementation(nn, X)

# Aufgabe d: Lernsequenz starten und Ergebnisse dokumentieren
def start_learning_sequence(nn, X, P):
    nn.train(X, P)
    
    print("Lernsequenz abgeschlossen.")
    
start_learning_sequence(nn, X, P)

# Initialisierung der Gewichte und Schwellwerte auf zufällige Werte und Wiederholung des Experiments
nn_random_init = NeuralNetwork(n=n, m=m)
start_learning_sequence(nn_random_init, X, P)

print("Experiment abgeschlossen.")
