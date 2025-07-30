import matplotlib.pyplot as plt

def plot_prediction_sequence(scores):
    plt.plot(scores, marker='o')
    plt.title("Content Safety Score over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Safety Confidence")
    plt.grid()
    plt.show()
