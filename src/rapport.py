"""
=============================================================
  Vision par Ordinateur — Génération du rapport
  Auteur : Hypollite Jean-Marc
=============================================================
Génère un fichier rapport.txt structuré selon le plan du projet.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_PATH = BASE_DIR / "images" / "sommet.jpeg"
RES_DIR  = BASE_DIR / "results"
RES_DIR.mkdir(exist_ok=True)


def generer_rapport():
    img_bgr  = cv2.imread(str(IMG_PATH))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w     = img_gray.shape

    lines = []
    add   = lines.append

    add("=" * 65)
    add("  VISION PAR ORDINATEUR — Master 2 BDGL")
    add("  Rapport de projet")
    add(f"  Auteur : Hypollite Jean-Marc")
    add(f"  Date   : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    add("=" * 65)

    # ── Introduction ──
    add("\n1. INTRODUCTION")
    add("-" * 40)
    add("Ce projet présente cinq méthodes classiques de vision par ordinateur")
    add("appliquées sur l'image de couverture du livre «Rendez-vous au sommet»")
    add("de Zig Ziglar. L'image contient du texte, des bords nets et des textures")
    add("variées — un cas de test riche pour les détecteurs de points d'intérêt.")
    add(f"\nImage : {IMG_PATH.name}  |  Dimensions : {w}×{h} px")

    # ── Présentation des méthodes ──
    add("\n2. PRÉSENTATION DES MÉTHODES")
    add("-" * 40)

    methods = {
        "Harris (1988)": (
            "Détecteur de coins basé sur la matrice de structure (matrice M = A^T A).\n"
            "  Réponse R = det(M) - k·trace(M)². Coin si R > seuil.\n"
            "  Avantages : simple, rapide, invariant à la rotation.\n"
            "  Limites   : sensible à l'échelle, non invariant au zoom."
        ),
        "SIFT (Lowe, 2004)": (
            "Détecte des keypoints stables dans l'espace-échelle (DoG).\n"
            "  Descripteur : histogramme de gradients 128-D.\n"
            "  Avantages : invariant à l'échelle, rotation, illumination.\n"
            "  Limites   : lent, breveté (domaine public depuis 2020)."
        ),
        "SURF (Bay, 2006)": (
            "Version accélérée de SIFT via filtres Haar approximés (Hessien).\n"
            "  Descripteur : 64-D ou 128-D.\n"
            "  Avantages : 3× plus rapide que SIFT.\n"
            "  Limites   : breveté (OpenCV contrib uniquement)."
        ),
        "ORB (Rublee, 2011)": (
            "Combine FAST (détection) + BRIEF orienté (description).\n"
            "  Descripteur binaire 256 bits — distance de Hamming.\n"
            "  Avantages : très rapide, libre de droits, embarqué.\n"
            "  Limites   : moins robuste que SIFT sur les grandes transformations."
        ),
        "GLCM / Haralick (1973)": (
            "Matrice de co-occurrence des niveaux de gris — texture statistique.\n"
            "  5 descripteurs : contraste, dissimilarité, homogénéité, énergie, corrélation.\n"
            "  Avantages : simple, interprétable, efficace pour la classification.\n"
            "  Limites   : ne capture pas la structure spatiale globale."
        ),
    }

    for name, desc in methods.items():
        add(f"\n  ► {name}")
        for line in desc.split("\n"):
            add(f"    {line}")

    # ── Matériels & méthodes ──
    add("\n3. MATÉRIELS ET MÉTHODES")
    add("-" * 40)
    add("  Matériels")
    add("    OS       : Ubuntu / Windows / macOS")
    add("    IDE      : Visual Studio Code")
    add("    Python   : 3.8+")
    add("    Librairies principales :")
    add("      • opencv-python   (Harris, SIFT, ORB)")
    add("      • scikit-image    (GLCM / Haralick)")
    add("      • numpy, matplotlib")
    add("\n  Données")
    add(f"    Image unique : {IMG_PATH.name}  ({w}×{h} px, RGB)")
    add("    Source : couverture du livre «Rendez-vous au sommet», Zig Ziglar")
    add("\n  Code")
    add("    src/main.py    — pipeline principal (Harris→SIFT→SURF→ORB→GLCM)")
    add("    src/rapport.py — génération de ce rapport")

    # ── Résultats ──
    add("\n4. RÉSULTATS ET DISCUSSION")
    add("-" * 40)

    # Harris
    gray_float = np.float32(img_gray)
    resp       = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    resp       = cv2.dilate(resp, None)
    n_corners  = int((resp > 0.01 * resp.max()).sum())
    add(f"  Harris   : {n_corners} coins détectés")

    # SIFT
    sift = cv2.SIFT_create(500)
    kps, _ = sift.detectAndCompute(img_gray, None)
    add(f"  SIFT     : {len(kps)} keypoints  (descripteurs 128-D)")

    # ORB
    orb = cv2.ORB_create(1000)
    kps_orb, _ = orb.detectAndCompute(img_gray, None)
    add(f"  ORB      : {len(kps_orb)} keypoints  (descripteurs 256-bits)")

    # SURF fallback
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        kps_surf, _ = surf.detectAndCompute(img_gray, None)
        add(f"  SURF     : {len(kps_surf)} keypoints")
    except (AttributeError, cv2.error):
        brisk = cv2.BRISK_create()
        kps_surf, _ = brisk.detectAndCompute(img_gray, None)
        add(f"  SURF     : algorithme breveté → BRISK utilisé en substitut ({len(kps_surf)} kp)")

    # GLCM
    gray_r = (img_gray // 4).astype(np.uint8)
    glcm   = graycomatrix(gray_r, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 64, symmetric=True, normed=True)
    add("\n  Descripteurs GLCM (moyenne sur 4 angles) :")
    for p in ["contrast","dissimilarity","homogeneity","energy","correlation"]:
        val = graycoprops(glcm, p).mean()
        add(f"    {p:16s}: {val:.4f}")

    add("\n  Discussion")
    add("  SIFT et SURF détectent des zones riches en gradient (lettres, bords du livre).")
    add("  ORB, plus rapide, donne des keypoints comparables mais moins précis.")
    add("  Harris est idéal pour les coins nets (coins du livre, lettres en relief).")
    add("  GLCM révèle une texture modérément homogène avec faible contraste global,")
    add("  ce qui est cohérent avec une image d'impression sur fond coloré uni.")

    # ── Conclusion ──
    add("\n5. CONCLUSION")
    add("-" * 40)
    add("  Ce projet a permis d'implémenter et comparer 5 méthodes fondamentales")
    add("  de vision par ordinateur sur une image réelle contenant texte et formes.")
    add("  Harris reste une référence pour les coins ; SIFT est le plus robuste en")
    add("  matching ; ORB est idéal pour les systèmes embarqués ; GLCM offre une")
    add("  description de texture utile pour la classification d'images.")
    add("\n  Résultats visuels : voir results/resultats_complets.png")
    add("=" * 65)

    report_path = RES_DIR / "rapport.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] Rapport sauvegardé → {report_path}")


if __name__ == "__main__":
    generer_rapport()
