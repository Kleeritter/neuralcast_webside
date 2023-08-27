import numpy as np
import matplotlib.pyplot as plt

# Beispielwerte für Beobachtungen und Modelle
observations = np.array([1, 2, 3, 4, 5])
model1 = np.array([0.8, 1.9, 2.7, 3.8, 5.2])
model2 = np.array([1.2, 1.8, 3.2, 4.2, 5.5])

# Berechnung der Korrelationskoeffizienten
correlation1 = np.corrcoef(observations, model1)[0, 1]
correlation2 = np.corrcoef(observations, model2)[0, 1]

# Taylor-Diagramm erstellen
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')

# Beobachtungen
ax.plot([0, correlation1], [0, np.std(observations)], 'ko', linestyle='dashed', label='Beobachtungen')

# Modell 1
ax.plot([0, correlation2], [0, np.std(model1)], 'bo', label='Modell 1')

# Achsenformatierung
ax.set_rmax(np.std(observations))
ax.set_xticks(np.arange(0, 2 * np.pi, np.pi/4))
ax.set_xticklabels(['0', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$', '$5\pi/4$', '$3\pi/2$', '$7\pi/4$'])
ax.set_yticklabels([])
ax.spines['polar'].set_visible(False)

# Legende
ax.legend(loc='upper right')

# Titel
plt.title('Taylor-Diagramm')

# Anzeigen
plt.show()
