# Vision par Ordinateur — Master 2 BDGL
## Harris · SIFT · SURF · ORB · GLCM
**Auteur :** Hypollite Jean-Marc

---

## Structure du projet
```
vision_projet/
├── images/
│   └── sommet.jpeg        ← image test
├── results/               ← figures et rapport (générés)
├── src/
│   ├── main.py            ← pipeline principal
│   └── rapport.py         ← génération du rapport texte
├── requirements.txt
└── README.md
```

## Installation (une seule fois)
```bash
pip install -r requirements.txt
```

## Lancer le projet
```bash
# Pipeline complet + figure résumé
python src/main.py

# Générer le rapport texte
python src/rapport.py
```

## Méthodes implémentées

| Méthode | Type | Descripteur | Invariant |
|---------|------|-------------|-----------|
| **Harris** | Détecteur de coins | — | Rotation |
| **SIFT** | Keypoints | 128-D flottant | Échelle + Rotation |
| **SURF** | Keypoints (rapide) | 64-D flottant | Échelle + Rotation |
| **ORB** | Keypoints (libre) | 256-bits binaire | Rotation |
| **GLCM** | Texture (Haralick) | 5 scalaires | — |

> **Note SURF :** SURF nécessite `opencv-contrib-python`.  
> Si absent, le script utilise automatiquement **BRISK** en substitut.

## Résultats attendus
- `results/resultats_complets.png` — figure 6 panneaux (image originale + 5 méthodes)
- `results/rapport.txt` — rapport structuré selon le plan du projet
