"""
=============================================================
  Vision par Ordinateur - Master 2 BDGL
  Méthodes : Harris | SIFT | SURF | ORB | GLCM
  Auteur    : Hypollite Jean-Marc
  Image test: "Rendez-vous au sommet" - Zig Ziglar
=============================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
IMG_PATH  = BASE_DIR / "images" / "sommet.jpeg"
RES_DIR   = BASE_DIR / "results"
RES_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  CHARGEMENT
# ═══════════════════════════════════════════════════════════
def load_image(path: Path):
    """Charge l'image en couleur ET en niveaux de gris."""
    img_bgr  = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image introuvable : {path}")
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(f"[OK] Image chargée  {img_rgb.shape}  {path.name}")
    return img_rgb, img_gray


# ═══════════════════════════════════════════════════════════
#  1. HARRIS  — Détecteur de coins
# ═══════════════════════════════════════════════════════════
def apply_harris(gray: np.ndarray, img_rgb: np.ndarray, k=0.04, threshold=0.01):
    """
    Détecte les coins avec l'algorithme de Harris & Stephens (1988).
    - blockSize : taille du voisinage analysé
    - ksize     : taille du noyau Sobel pour le gradient
    - k         : paramètre libre (~0.04–0.06)
    """
    gray_float = np.float32(gray)

    # Réponse Harris (R = det(M) - k * trace(M)²)
    harris_resp = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=k)

    # Dilatation pour marquer tous les pixels d'un coin
    harris_resp = cv2.dilate(harris_resp, None)

    # Seuillage
    img_result = img_rgb.copy()
    corners    = harris_resp > threshold * harris_resp.max()
    img_result[corners] = [255, 0, 0]          # rouge

    n_corners = int(corners.sum())
    print(f"[Harris] {n_corners} coins détectés  (k={k}, seuil={threshold})")
    return img_result, n_corners


# ═══════════════════════════════════════════════════════════
#  2. SIFT  — Scale-Invariant Feature Transform
# ═══════════════════════════════════════════════════════════
def apply_sift(gray: np.ndarray, img_rgb: np.ndarray, n_features=500):
    """
    Extrait des descripteurs invariants à l'échelle et à la rotation.
    Chaque keypoint → descripteur 128-D.
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    kps, descriptors = sift.detectAndCompute(gray, None)

    img_result = cv2.drawKeypoints(
        img_rgb, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 200, 0)
    )

    print(f"[SIFT]  {len(kps)} keypoints  |  descripteurs : {descriptors.shape if descriptors is not None else 'None'}")
    return img_result, kps, descriptors


# ═══════════════════════════════════════════════════════════
#  3. SURF  — Speeded-Up Robust Features
# ═══════════════════════════════════════════════════════════
def apply_surf_or_fallback(gray: np.ndarray, img_rgb: np.ndarray):
    """
    SURF est breveté → disponible seulement avec opencv-contrib.
    Si absent, on utilise BRISK (même famille, open-source) comme
    substitut pédagogique et on l'indique clairement.
    """
    try:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        kps, descriptors = surf.detectAndCompute(gray, None)
        method_name = "SURF"
    except (AttributeError, cv2.error):
        print("[SURF]  Module xfeatures2d non disponible → fallback BRISK")
        surf = cv2.BRISK_create()
        kps, descriptors = surf.detectAndCompute(gray, None)
        method_name = "BRISK (≈SURF)"

    img_result = cv2.drawKeypoints(
        img_rgb, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 100, 255)
    )

    desc_shape = descriptors.shape if descriptors is not None else "None"
    print(f"[{method_name}] {len(kps)} keypoints  |  descripteurs : {desc_shape}")
    return img_result, kps, descriptors, method_name


# ═══════════════════════════════════════════════════════════
#  4. ORB  — Oriented FAST + Rotated BRIEF
# ═══════════════════════════════════════════════════════════
def apply_orb(gray: np.ndarray, img_rgb: np.ndarray, n_features=1000):
    """
    Alternative libre à SIFT/SURF.  Rapide, binaire (Hamming distance).
    Descripteur 256-bits.
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    kps, descriptors = orb.detectAndCompute(gray, None)

    img_result = cv2.drawKeypoints(
        img_rgb, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(255, 165, 0)
    )

    desc_shape = descriptors.shape if descriptors is not None else "None"
    print(f"[ORB]   {len(kps)} keypoints  |  descripteurs : {desc_shape}")
    return img_result, kps, descriptors


# ═══════════════════════════════════════════════════════════
#  5. GLCM  — Gray-Level Co-occurrence Matrix (Haralick)
# ═══════════════════════════════════════════════════════════
def apply_glcm(gray: np.ndarray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Calcule la matrice de co-occurrence et en extrait 5 descripteurs
    de texture de Haralick :
      contrast, dissimilarity, homogeneity, energy, correlation
    """
    # Réduction à 64 niveaux pour la lisibilité
    gray_reduced = (gray // 4).astype(np.uint8)

    glcm = graycomatrix(
        gray_reduced,
        distances=distances,
        angles=angles,
        levels=64,
        symmetric=True,
        normed=True
    )

    props = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        props[prop] = graycoprops(glcm, prop)

    print("[GLCM]  Descripteurs de texture extraits :")
    for k, v in props.items():
        print(f"         {k:15s} → {v.mean():.4f}")

    # Visualisation de la GLCM (première distance, angle 0°)
    glcm_vis = (glcm[:, :, 0, 0] * 255 / glcm[:, :, 0, 0].max()).astype(np.uint8)
    return glcm_vis, props


# ═══════════════════════════════════════════════════════════
#  FIGURE RÉSUMÉ
# ═══════════════════════════════════════════════════════════
def build_figure(img_rgb, gray,
                 harris_img, n_corners,
                 sift_img, sift_kps,
                 surf_img, surf_kps, surf_name,
                 orb_img, orb_kps,
                 glcm_vis, glcm_props):

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#1e1e2e")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    axes_cfg = [
        (img_rgb,   "Image originale",             gs[0, 0], False),
        (harris_img,f"Harris  ({n_corners} coins)", gs[0, 1], False),
        (sift_img,  f"SIFT  ({len(sift_kps)} kp)", gs[0, 2], False),
        (surf_img,  f"{surf_name}  ({len(surf_kps)} kp)", gs[1, 0], False),
        (orb_img,   f"ORB  ({len(orb_kps)} kp)",   gs[1, 1], False),
        (glcm_vis,  "GLCM  (matrice de co-occ.)",  gs[1, 2], True),
    ]

    for img, title, pos, is_gray in axes_cfg:
        ax = fig.add_subplot(pos)
        if is_gray:
            ax.imshow(img, cmap="hot")
        else:
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

    # Tableau GLCM
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.set_facecolor("#1e1e2e")
    ax_table.axis("off")

    angle_labels = ["0°", "45°", "90°", "135°"]
    cols = ["Propriété"] + angle_labels + ["Moyenne"]
    rows = []
    for prop, values in glcm_props.items():
        row = [prop.capitalize()] + [f"{v:.4f}" for v in values[0]] + [f"{values.mean():.4f}"]
        rows.append(row)

    table = ax_table.table(
        cellText=rows, colLabels=cols,
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style de la table
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#2a2a3e" if r == 0 else ("#313145" if r % 2 == 0 else "#252535"))
        cell.set_text_props(color="white")
        cell.set_edgecolor("#555580")

    ax_table.set_title(
        "Descripteurs GLCM (Haralick) — distance=1, 4 angles",
        color="white", fontsize=12, fontweight="bold", pad=10
    )

    fig.suptitle(
        "Vision par Ordinateur — Harris · SIFT · SURF · ORB · GLCM\n"
        "Image : «Rendez-vous au sommet» de Zig Ziglar",
        color="white", fontsize=14, fontweight="bold", y=0.98
    )

    out_path = RES_DIR / "resultats_complets.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[OK] Figure sauvegardée → {out_path}")
    plt.show(block=False)
    input("Appuie sur Entrée pour fermer...")
    plt.close()


# ═══════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Vision par Ordinateur — Master 2 BDGL")
    print("=" * 55)

    img_rgb, gray = load_image(IMG_PATH)

    print("\n── 1. Harris ─────────────────────────────────────")
    harris_img, n_corners = apply_harris(gray, img_rgb)

    print("\n── 2. SIFT ───────────────────────────────────────")
    sift_img, sift_kps, sift_desc = apply_sift(gray, img_rgb)

    print("\n── 3. SURF ───────────────────────────────────────")
    surf_img, surf_kps, surf_desc, surf_name = apply_surf_or_fallback(gray, img_rgb)

    print("\n── 4. ORB ────────────────────────────────────────")
    orb_img, orb_kps, orb_desc = apply_orb(gray, img_rgb)

    print("\n── 5. GLCM ───────────────────────────────────────")
    glcm_vis, glcm_props = apply_glcm(gray)

    print("\n── Génération de la figure résumé ────────────────")
    build_figure(
        img_rgb, gray,
        harris_img, n_corners,
        sift_img, sift_kps,
        surf_img, surf_kps, surf_name,
        orb_img, orb_kps,
        glcm_vis, glcm_props
    )

    print("\n[FIN]  Tous les résultats sont dans /results/")
