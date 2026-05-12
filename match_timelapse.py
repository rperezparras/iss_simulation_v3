#!/usr/bin/env python3
"""
Empareja y alinea imágenes simuladas (output) y reales (pics) de un timelapse ISS.

- Usa matcher ('superpoint-lg') si está disponible.
- Extrae matches aunque cambien las keys del dict devuelto.
- Fallback a ORB+RANSAC si el matcher devuelve 0 matches.
- Ajusta PolynomialTransform entre real y simulada.
- Genera rejilla en real y la proyecta a simulada.
- Guarda CSV con pares (sim_x, sim_y, real_x, real_y).
"""

import os
import sys
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageFile, ImageOps
import torchvision.transforms as tfm

from skimage.metrics import structural_similarity as ssim
from skimage.transform import PolynomialTransform

import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------------------
# Helpers
# ----------------------------

def invert_y_coordinate(y: float, height: int) -> float:
    """Convierte y (origen arriba) -> y' (origen abajo)."""
    return float(height) - float(y)

def to_numpy(x):
    """Convierte torch/numpy/list a np.ndarray."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        x = x.detach()
        if x.is_cuda:
            x = x.cpu()
        return x.numpy()
    return np.array(x)

def squeeze_points(arr: np.ndarray) -> np.ndarray:
    """Asegura shape (N,2)."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    # Ej: (1,N,2) -> (N,2)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr

def image_loader(path: str, resize=None) -> torch.Tensor:
    """
    Carga una imagen como tensor float32 en [0,1], shape (3,H,W).
    Aplica EXIF transpose.
    """
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        # Fallback con cv2 si PIL se queja
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

    if resize is not None:
        # resize = (H, W)
        img = img.resize((int(resize[1]), int(resize[0])), resample=Image.BILINEAR)

    tens = tfm.ToTensor()(img)  # (3,H,W), float32 [0,1]
    return tens


def extract_mkpts_from_result(result: dict):
    """
    Soporta múltiples formatos de salida del matcher.

    Devuelve:
      mkpts0 (simulada) np.ndarray (N,2)
      mkpts1 (real)     np.ndarray (N,2)
    """
    if result is None:
        return None, None

    # Caso clásico (como tu versión “buena”)
    for k0, k1 in [
        ("mkpts0", "mkpts1"),
        ("m_kpts0", "m_kpts1"),
        ("matched_kpts0", "matched_kpts1"),
    ]:
        if k0 in result and k1 in result:
            a = squeeze_points(to_numpy(result[k0]))
            b = squeeze_points(to_numpy(result[k1]))
            return a, b

    # Otros nombres habituales
    for k0, k1 in [
        ("kpts0", "kpts1"),
        ("keypoints0", "keypoints1"),
    ]:
        if k0 in result and k1 in result:
            pts0 = squeeze_points(to_numpy(result[k0]))
            pts1 = squeeze_points(to_numpy(result[k1]))

            # matches Nx2
            if "matches" in result:
                m = to_numpy(result["matches"])
                if m is not None:
                    m = np.asarray(m).astype(np.int64)
                    if m.ndim == 2 and m.shape[1] == 2:
                        mk0 = pts0[m[:, 0]]
                        mk1 = pts1[m[:, 1]]
                        return mk0, mk1

            # matches0: array (N0,) con índice en pts1 o -1
            if "matches0" in result:
                m0 = to_numpy(result["matches0"])
                if m0 is not None:
                    m0 = np.asarray(m0).astype(np.int64)
                    valid = m0 >= 0
                    mk0 = pts0[valid]
                    mk1 = pts1[m0[valid]]
                    return mk0, mk1

            # matches1: array (N1,) con índice en pts0 o -1
            if "matches1" in result:
                m1 = to_numpy(result["matches1"])
                if m1 is not None:
                    m1 = np.asarray(m1).astype(np.int64)
                    valid = m1 >= 0
                    mk1 = pts1[valid]
                    mk0 = pts0[m1[valid]]
                    return mk0, mk1

    return None, None


def orb_fallback_matches(img0_bgr, img1_bgr, max_matches=2000):
    """
    Fallback CPU: ORB + ratio test + RANSAC (homography) para filtrar inliers.
    Devuelve mkpts0 (sim), mkpts1 (real) en píxeles.
    """
    g0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=8000, fastThreshold=7)
    kp0, des0 = orb.detectAndCompute(g0, None)
    kp1, des1 = orb.detectAndCompute(g1, None)

    if des0 is None or des1 is None or len(kp0) < 10 or len(kp1) < 10:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des0, des1, k=2)

    good = []
    for a, b in knn:
        if a.distance < 0.75 * b.distance:
            good.append(a)

    if len(good) < 20:
        return None, None

    # ordenar por distancia y recortar
    good = sorted(good, key=lambda m: m.distance)[:max_matches]

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])  # sim
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])  # real

    # Filtrado RANSAC con homografía (real->sim)
    H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 3.0)
    if mask is None:
        return None, None

    mask = mask.ravel().astype(bool)
    pts0_in = pts0[mask]
    pts1_in = pts1[mask]

    if len(pts0_in) < 20:
        return None, None

    return pts0_in, pts1_in


def normalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    out = np.empty_like(points, dtype=np.float64)
    out[:, 0] = 2.0 * (points[:, 0] / max(width - 1, 1)) - 1.0
    out[:, 1] = 2.0 * (points[:, 1] / max(height - 1, 1)) - 1.0
    return out


def denormalize_points(points_n: np.ndarray, width: int, height: int) -> np.ndarray:
    points_n = np.asarray(points_n, dtype=np.float64)
    out = np.empty_like(points_n, dtype=np.float64)
    out[:, 0] = 0.5 * (points_n[:, 0] + 1.0) * max(width - 1, 1)
    out[:, 1] = 0.5 * (points_n[:, 1] + 1.0) * max(height - 1, 1)
    return out


def apply_polynomial_transform(
    matches_src,
    matches_dst,
    grid_points,
    src_width,
    src_height,
    dst_width,
    dst_height,
    order=2,
):
    src_n = normalize_points(matches_src, src_width, src_height)
    dst_n = normalize_points(matches_dst, dst_width, dst_height)
    grid_n = normalize_points(grid_points, src_width, src_height)

    poly_transform = PolynomialTransform()
    ok = poly_transform.estimate(src_n, dst_n, order=order)

    if not ok:
        return None, None

    transformed_n = poly_transform(grid_n)

    # Aquí vuelves a coordenadas píxel simuladas
    transformed_points = denormalize_points(
        transformed_n,
        dst_width,
        dst_height,
    )

    return poly_transform, transformed_points


def generate_grid_points(img_width, img_height, step=100):
    points = []
    for y in range(0, img_height, step):
        for x in range(0, img_width, step):
            points.append((x, y))
    return np.array(points)


def draw_matches(img1, img2, mkpts0, mkpts1, title, save_path=None):
    img_combined = np.concatenate((img1, img2), axis=1)
    offset = img1.shape[1]
    mkpts1_offset = [(pt[0] + offset, pt[1]) for pt in mkpts1]

    for pt0, pt1 in zip(mkpts0, mkpts1_offset):
        pt0 = tuple(map(int, pt0))
        pt1 = tuple(map(int, pt1))
        cv2.line(img_combined, pt0, pt1, color=(0, 255, 0), thickness=2)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ----------------------------
# MAIN
# ----------------------------

def main(args: argparse.Namespace | None = None):
    if args is None:
        parser = argparse.ArgumentParser(description="Matching ISS simulado-real")
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--pictures_dir", type=str, required=True)
        parser.add_argument("--matches_output_dir", type=str, required=True)
        parser.add_argument("--grid_step", type=int, default=265)
        parser.add_argument("--show_every", type=int, default=100)

        # NUEVO (opcionales, no rompen pipeline)
        parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
        parser.add_argument("--matching_repo", type=str, default=None,
                            help="Ruta a image-matching-models (si no, intenta autodetectar).")
        parser.add_argument("--min_matches", type=int, default=20)
        parser.add_argument("--no_orb_fallback", action="store_true")
        args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    pictures_dir = Path(args.pictures_dir).expanduser()
    matches_output_dir = Path(args.matches_output_dir).expanduser()
    matches_output_dir.mkdir(parents=True, exist_ok=True)

    output_files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("render_output_") and f.lower().endswith(".png")
    )
    pictures_files = sorted(
        f for f in os.listdir(pictures_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    )

    if not output_files:
        print(f"No se encontraron simuladas en {output_dir}")
        return
    if not pictures_files:
        print(f"No se encontraron reales en {pictures_dir}")
        return

    print(f"Nº de simuladas: {len(output_files)}")
    print(f"Nº de reales:    {len(pictures_files)}")

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA no disponible, usando CPU.")
            device = "cpu"

    print(f"torch: {torch.__version__} | device: {device}")

    # Import matcher (robusto a rutas)
    matching_repo = args.matching_repo
    if matching_repo is None:
        # Autodetect rápido
        candidates = [
            os.environ.get("IMAGE_MATCHING_MODELS_DIR", ""),
            "/home/rpz/image-matching-models",
            "/home/raul/image-matching-models",
        ]
        for c in candidates:
            if c and Path(c).exists():
                matching_repo = c
                break

    if matching_repo and Path(matching_repo).exists():
        sys.path.insert(0, matching_repo)

    try:
        from matching import get_matcher  # type: ignore
        import matching as matching_mod  # type: ignore
        print(f"matching importado desde: {matching_mod.__file__}")
        matcher = get_matcher("superpoint-lg", device=device)
    except Exception as e:
        matcher = None
        print("⚠️ No se pudo cargar el matcher superpoint-lg. Razón:", repr(e))
        print("   → Se usará ORB fallback en todos los frames (más lento).")

    # Loop pares (por orden)
    for idx, (img0_file, img1_file) in enumerate(zip(output_files, pictures_files)):
        img0_path = output_dir / img0_file   # simulada
        img1_path = pictures_dir / img1_file # real

        img0_bgr = cv2.imread(str(img0_path), cv2.IMREAD_COLOR)
        img1_bgr = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
        if img0_bgr is None or img1_bgr is None:
            continue

        h0, w0 = img0_bgr.shape[:2]
        h1, w1 = img1_bgr.shape[:2]

        mkpts0 = mkpts1 = None

        # ---- 1) Intento matcher
        if matcher is not None:
            img0_t = image_loader(str(img0_path)).to(device)
            img1_t = image_loader(str(img1_path)).to(device)

            # Algunos wrappers esperan (3,H,W), otros (1,3,H,W).
            # Probamos primero sin batch, si falla probamos con batch.
            with torch.inference_mode():
                try:
                    result = matcher(img0_t, img1_t)
                except Exception:
                    result = matcher(img0_t.unsqueeze(0), img1_t.unsqueeze(0))

            mkpts0, mkpts1 = extract_mkpts_from_result(result)

        n_m = 0 if mkpts0 is None else len(mkpts0)

        # ---- 2) Fallback ORB si no hay matches
        if (mkpts0 is None or mkpts1 is None or n_m < args.min_matches) and not args.no_orb_fallback:
            mkpts0_fb, mkpts1_fb = orb_fallback_matches(img0_bgr, img1_bgr)
            if mkpts0_fb is not None and len(mkpts0_fb) >= args.min_matches:
                mkpts0, mkpts1 = mkpts0_fb, mkpts1_fb
                n_m = len(mkpts0)
                print(f"🟡 Frame {idx}: usando ORB fallback ({n_m} inliers)")
            else:
                print(f"⚠️ Frame {idx}: matches insuficientes ({n_m}). No se genera CSV.")
                continue

        if mkpts0 is None or mkpts1 is None or len(mkpts0) < args.min_matches:
            print(f"⚠️ Frame {idx}: matches insuficientes ({0 if mkpts0 is None else len(mkpts0)}). No se genera CSV.")
            continue

        mkpts0 = np.asarray(mkpts0, dtype=np.float32)
        mkpts1 = np.asarray(mkpts1, dtype=np.float32)

        # ---- Filtrado RANSAC también para matches del matcher principal
        H, mask = cv2.findHomography(
            mkpts1.astype(np.float32),  # real
            mkpts0.astype(np.float32),  # sim
            cv2.RANSAC,
            3.0,
        )

        if mask is None:
            print(f"⚠️ Frame {idx}: RANSAC falló. No se genera CSV.")
            continue

        n_before = len(mkpts0)

        inliers = mask.ravel().astype(bool)
        mkpts0 = mkpts0[inliers]
        mkpts1 = mkpts1[inliers]

        n_after = len(mkpts0)
        print(
            f"Frame {idx}: matches={n_before}, "
            f"inliers={n_after}, "
            f"ratio={n_after / max(n_before, 1):.2f}"
        )

        if len(mkpts0) < args.min_matches:
            print(f"⚠️ Frame {idx}: pocos inliers tras RANSAC ({len(mkpts0)}). No se genera CSV.")
            continue

        # ---- Grid en real
        real_points = generate_grid_points(w1, h1, step=args.grid_step)

        # ---- Transform real->sim (src=real, dst=sim)
        poly_transform, transformed_grid_points = apply_polynomial_transform(
            matches_src=mkpts1,
            matches_dst=mkpts0,
            grid_points=real_points,
            src_width=w1,
            src_height=h1,
            dst_width=w0,
            dst_height=h0,
            order=2,
        )

        if poly_transform is None or transformed_grid_points is None:
            print(f"⚠️ Frame {idx}: falló el ajuste polinómico. No se genera CSV.")
            continue

        inside = (
            (transformed_grid_points[:, 0] >= 0) &
            (transformed_grid_points[:, 0] < w0) &
            (transformed_grid_points[:, 1] >= 0) &
            (transformed_grid_points[:, 1] < h0)
        )

        real_points_valid = real_points[inside]
        transformed_grid_points_valid = transformed_grid_points[inside]

        if len(real_points_valid) < args.min_matches:
            print(
                f"⚠️ Frame {idx}: pocos puntos válidos tras transformar "
                f"({len(real_points_valid)}). No se genera CSV."
            )
            continue

        # ---- Guardar CSV
        csv_filename = f"transformed_coordinates_{os.path.splitext(img0_file)[0]}.csv"
        csv_path = matches_output_dir / csv_filename

        with open(csv_path, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sim_x", "sim_y", "real_x", "real_y"])

            for pt0, pt1 in zip(transformed_grid_points_valid, real_points_valid):
                sim_x = float(pt0[0])
                sim_y = float(invert_y_coordinate(pt0[1], h0))
                real_x = float(pt1[0])
                real_y = float(invert_y_coordinate(pt1[1], h1))
                w.writerow([sim_x, sim_y, real_x, real_y])

        # ---- Plots cada N
        if idx % args.show_every == 0:
            # métricas rápidas
            img0_gray = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
            img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
            try:
                ssim_value = ssim(img1_gray, img0_gray)
            except Exception:
                ssim_value = None
            print(f"Procesando par {idx}: {img0_file} vs {img1_file} | matches={len(mkpts0)} | SSIM={ssim_value}")

            match_img_path = matches_output_dir / f"matches_{idx}.png"
            draw_matches(img0_bgr, img1_bgr, mkpts0, mkpts1,
                         f"Matches sim-real (idx={idx})", save_path=str(match_img_path))

    print("Proceso terminado para todos los pares.")


if __name__ == "__main__":
    main()
